#include <iostream>
#include <unordered_map>
#include "cublas_api.h"

#include "hook_api.h"
#include "../rpc.h"
extern void *(*real_dlsym)(void *, const char *);

extern "C" void *mem2server(void *clientPtr, size_t size);
extern "C" void mem2client(void *clientPtr, size_t size);
void *get_so_handle(const std::string &so_file);
int getSizeFromCudaDataType(cudaDataType type);
extern "C" cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
#ifdef DEBUG
    std::cout << "Hook: cublasCreate_v2 called" << std::endl;
#endif
    void *_0handle = mem2server((void *)handle, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCreate_v2);
    rpc_write(client, &_0handle, sizeof(_0handle));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)handle, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
#ifdef DEBUG
    std::cout << "Hook: cublasDestroy_v2 called" << std::endl;
#endif
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDestroy_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetVersion_v2 called" << std::endl;
#endif
    void *_0version = mem2server((void *)version, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetVersion_v2);
    rpc_write(client, &handle, sizeof(handle));
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

extern "C" cublasStatus_t cublasGetProperty(libraryPropertyType type, int *value) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetProperty called" << std::endl;
#endif
    void *_0value = mem2server((void *)value, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetProperty);
    rpc_write(client, &type, sizeof(type));
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

extern "C" size_t cublasGetCudartVersion() {
#ifdef DEBUG
    std::cout << "Hook: cublasGetCudartVersion called" << std::endl;
#endif
    size_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetCudartVersion);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetWorkspace_v2 called" << std::endl;
#endif
    void *_0workspace = mem2server((void *)workspace, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSetWorkspace_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &_0workspace, sizeof(_0workspace));
    rpc_write(client, &workspaceSizeInBytes, sizeof(workspaceSizeInBytes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)workspace, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetStream_v2 called" << std::endl;
#endif
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSetStream_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &streamId, sizeof(streamId));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *streamId) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetStream_v2 called" << std::endl;
#endif
    void *_0streamId = mem2server((void *)streamId, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetStream_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &_0streamId, sizeof(_0streamId));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)streamId, 0);
    return _result;
}

extern "C" cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetPointerMode_v2 called" << std::endl;
#endif
    void *_0mode = mem2server((void *)mode, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetPointerMode_v2);
    rpc_write(client, &handle, sizeof(handle));
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

extern "C" cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetPointerMode_v2 called" << std::endl;
#endif
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSetPointerMode_v2);
    rpc_write(client, &handle, sizeof(handle));
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

extern "C" cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetAtomicsMode called" << std::endl;
#endif
    void *_0mode = mem2server((void *)mode, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetAtomicsMode);
    rpc_write(client, &handle, sizeof(handle));
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

extern "C" cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetAtomicsMode called" << std::endl;
#endif
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSetAtomicsMode);
    rpc_write(client, &handle, sizeof(handle));
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

extern "C" cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetMathMode called" << std::endl;
#endif
    void *_0mode = mem2server((void *)mode, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetMathMode);
    rpc_write(client, &handle, sizeof(handle));
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

extern "C" cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetMathMode called" << std::endl;
#endif
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSetMathMode);
    rpc_write(client, &handle, sizeof(handle));
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

extern "C" cublasStatus_t cublasGetSmCountTarget(cublasHandle_t handle, int *smCountTarget) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetSmCountTarget called" << std::endl;
#endif
    void *_0smCountTarget = mem2server((void *)smCountTarget, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetSmCountTarget);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &_0smCountTarget, sizeof(_0smCountTarget));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)smCountTarget, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetSmCountTarget called" << std::endl;
#endif
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSetSmCountTarget);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &smCountTarget, sizeof(smCountTarget));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" const char *cublasGetStatusName(cublasStatus_t status) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetStatusName called" << std::endl;
#endif
    char *_cublasGetStatusName_result = nullptr;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetStatusName);
    rpc_write(client, &status, sizeof(status));
    rpc_read(client, &_cublasGetStatusName_result, 0, true);
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _cublasGetStatusName_result;
}

extern "C" const char *cublasGetStatusString(cublasStatus_t status) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetStatusString called" << std::endl;
#endif
    char *_cublasGetStatusString_result = nullptr;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetStatusString);
    rpc_write(client, &status, sizeof(status));
    rpc_read(client, &_cublasGetStatusString_result, 0, true);
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _cublasGetStatusString_result;
}

extern "C" cublasStatus_t cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char *logFileName) {
#ifdef DEBUG
    std::cout << "Hook: cublasLoggerConfigure called" << std::endl;
#endif
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasLoggerConfigure);
    rpc_write(client, &logIsOn, sizeof(logIsOn));
    rpc_write(client, &logToStdOut, sizeof(logToStdOut));
    rpc_write(client, &logToStdErr, sizeof(logToStdErr));
    rpc_write(client, logFileName, strlen(logFileName) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cublasStatus_t cublasSetLoggerCallback(cublasLogCallback userCallback) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetLoggerCallback called" << std::endl;
#endif
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSetLoggerCallback);
    rpc_write(client, &userCallback, sizeof(userCallback));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cublasStatus_t cublasGetLoggerCallback(cublasLogCallback *userCallback) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetLoggerCallback called" << std::endl;
#endif
    void *_0userCallback = mem2server((void *)userCallback, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetLoggerCallback);
    rpc_write(client, &_0userCallback, sizeof(_0userCallback));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)userCallback, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSetVector(int n, int elemSize, const void *x, int incx, void *devicePtr, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetVector called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0devicePtr = mem2server((void *)devicePtr, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSetVector);
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0devicePtr, sizeof(_0devicePtr));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)devicePtr, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSetVector_64(int64_t n, int64_t elemSize, const void *x, int64_t incx, void *devicePtr, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetVector_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0devicePtr = mem2server((void *)devicePtr, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSetVector_64);
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0devicePtr, sizeof(_0devicePtr));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)devicePtr, 0);
    return _result;
}

extern "C" cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetVector called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetVector);
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasGetVector_64(int64_t n, int64_t elemSize, const void *x, int64_t incx, void *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetVector_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetVector_64);
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetMatrix called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSetMatrix);
    rpc_write(client, &rows, sizeof(rows));
    rpc_write(client, &cols, sizeof(cols));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize, const void *A, int64_t lda, void *B, int64_t ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetMatrix_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSetMatrix_64);
    rpc_write(client, &rows, sizeof(rows));
    rpc_write(client, &cols, sizeof(cols));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetMatrix called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetMatrix);
    rpc_write(client, &rows, sizeof(rows));
    rpc_write(client, &cols, sizeof(cols));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" cublasStatus_t cublasGetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize, const void *A, int64_t lda, void *B, int64_t ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetMatrix_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetMatrix_64);
    rpc_write(client, &rows, sizeof(rows));
    rpc_write(client, &cols, sizeof(cols));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetVectorAsync called" << std::endl;
#endif
    void *_0hostPtr = mem2server((void *)hostPtr, 0);
    void *_0devicePtr = mem2server((void *)devicePtr, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSetVectorAsync);
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0hostPtr, sizeof(_0hostPtr));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0devicePtr, sizeof(_0devicePtr));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)hostPtr, 0);
    mem2client((void *)devicePtr, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSetVectorAsync_64(int64_t n, int64_t elemSize, const void *hostPtr, int64_t incx, void *devicePtr, int64_t incy, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetVectorAsync_64 called" << std::endl;
#endif
    void *_0hostPtr = mem2server((void *)hostPtr, 0);
    void *_0devicePtr = mem2server((void *)devicePtr, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSetVectorAsync_64);
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0hostPtr, sizeof(_0hostPtr));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0devicePtr, sizeof(_0devicePtr));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)hostPtr, 0);
    mem2client((void *)devicePtr, 0);
    return _result;
}

extern "C" cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetVectorAsync called" << std::endl;
#endif
    void *_0devicePtr = mem2server((void *)devicePtr, 0);
    void *_0hostPtr = mem2server((void *)hostPtr, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetVectorAsync);
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0devicePtr, sizeof(_0devicePtr));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0hostPtr, sizeof(_0hostPtr));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)devicePtr, 0);
    mem2client((void *)hostPtr, 0);
    return _result;
}

extern "C" cublasStatus_t cublasGetVectorAsync_64(int64_t n, int64_t elemSize, const void *devicePtr, int64_t incx, void *hostPtr, int64_t incy, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetVectorAsync_64 called" << std::endl;
#endif
    void *_0devicePtr = mem2server((void *)devicePtr, 0);
    void *_0hostPtr = mem2server((void *)hostPtr, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetVectorAsync_64);
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0devicePtr, sizeof(_0devicePtr));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0hostPtr, sizeof(_0hostPtr));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)devicePtr, 0);
    mem2client((void *)hostPtr, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetMatrixAsync called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSetMatrixAsync);
    rpc_write(client, &rows, sizeof(rows));
    rpc_write(client, &cols, sizeof(cols));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSetMatrixAsync_64(int64_t rows, int64_t cols, int64_t elemSize, const void *A, int64_t lda, void *B, int64_t ldb, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cublasSetMatrixAsync_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSetMatrixAsync_64);
    rpc_write(client, &rows, sizeof(rows));
    rpc_write(client, &cols, sizeof(cols));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetMatrixAsync called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetMatrixAsync);
    rpc_write(client, &rows, sizeof(rows));
    rpc_write(client, &cols, sizeof(cols));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" cublasStatus_t cublasGetMatrixAsync_64(int64_t rows, int64_t cols, int64_t elemSize, const void *A, int64_t lda, void *B, int64_t ldb, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cublasGetMatrixAsync_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGetMatrixAsync_64);
    rpc_write(client, &rows, sizeof(rows));
    rpc_write(client, &cols, sizeof(cols));
    rpc_write(client, &elemSize, sizeof(elemSize));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" void cublasXerbla(const char *srName, int info) {
#ifdef DEBUG
    std::cout << "Hook: cublasXerbla called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasXerbla);
    rpc_write(client, srName, strlen(srName) + 1, true);
    rpc_write(client, &info, sizeof(info));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return;
}

extern "C" cublasStatus_t cublasNrm2Ex(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, void *result, cudaDataType resultType, cudaDataType executionType) {
#ifdef DEBUG
    std::cout << "Hook: cublasNrm2Ex called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0result = mem2server((void *)result, n * getSizeFromCudaDataType(resultType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasNrm2Ex);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_write(client, &resultType, sizeof(resultType));
    rpc_write(client, &executionType, sizeof(executionType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)result, n * getSizeFromCudaDataType(resultType));
    return _result;
}

extern "C" cublasStatus_t cublasNrm2Ex_64(cublasHandle_t handle, int64_t n, const void *x, cudaDataType xType, int64_t incx, void *result, cudaDataType resultType, cudaDataType executionType) {
#ifdef DEBUG
    std::cout << "Hook: cublasNrm2Ex_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0result = mem2server((void *)result, n * getSizeFromCudaDataType(resultType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasNrm2Ex_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_write(client, &resultType, sizeof(resultType));
    rpc_write(client, &executionType, sizeof(executionType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)result, n * getSizeFromCudaDataType(resultType));
    return _result;
}

extern "C" cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasSnrm2_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSnrm2_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSnrm2_v2_64(cublasHandle_t handle, int64_t n, const float *x, int64_t incx, float *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasSnrm2_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSnrm2_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasDnrm2_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDnrm2_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDnrm2_v2_64(cublasHandle_t handle, int64_t n, const double *x, int64_t incx, double *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasDnrm2_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDnrm2_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasScnrm2_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, float *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasScnrm2_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasScnrm2_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasScnrm2_v2_64(cublasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, float *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasScnrm2_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasScnrm2_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDznrm2_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasDznrm2_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDznrm2_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDznrm2_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, double *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasDznrm2_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDznrm2_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDotEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, const void *y, cudaDataType yType, int incy, void *result, cudaDataType resultType, cudaDataType executionType) {
#ifdef DEBUG
    std::cout << "Hook: cublasDotEx called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0y = mem2server((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    void *_0result = mem2server((void *)result, n * getSizeFromCudaDataType(resultType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDotEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &yType, sizeof(yType));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_write(client, &resultType, sizeof(resultType));
    rpc_write(client, &executionType, sizeof(executionType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    mem2client((void *)result, n * getSizeFromCudaDataType(resultType));
    return _result;
}

extern "C" cublasStatus_t cublasDotEx_64(cublasHandle_t handle, int64_t n, const void *x, cudaDataType xType, int64_t incx, const void *y, cudaDataType yType, int64_t incy, void *result, cudaDataType resultType, cudaDataType executionType) {
#ifdef DEBUG
    std::cout << "Hook: cublasDotEx_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0y = mem2server((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    void *_0result = mem2server((void *)result, n * getSizeFromCudaDataType(resultType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDotEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &yType, sizeof(yType));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_write(client, &resultType, sizeof(resultType));
    rpc_write(client, &executionType, sizeof(executionType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    mem2client((void *)result, n * getSizeFromCudaDataType(resultType));
    return _result;
}

extern "C" cublasStatus_t cublasDotcEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, const void *y, cudaDataType yType, int incy, void *result, cudaDataType resultType, cudaDataType executionType) {
#ifdef DEBUG
    std::cout << "Hook: cublasDotcEx called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0y = mem2server((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    void *_0result = mem2server((void *)result, n * getSizeFromCudaDataType(resultType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDotcEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &yType, sizeof(yType));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_write(client, &resultType, sizeof(resultType));
    rpc_write(client, &executionType, sizeof(executionType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    mem2client((void *)result, n * getSizeFromCudaDataType(resultType));
    return _result;
}

extern "C" cublasStatus_t cublasDotcEx_64(cublasHandle_t handle, int64_t n, const void *x, cudaDataType xType, int64_t incx, const void *y, cudaDataType yType, int64_t incy, void *result, cudaDataType resultType, cudaDataType executionType) {
#ifdef DEBUG
    std::cout << "Hook: cublasDotcEx_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0y = mem2server((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    void *_0result = mem2server((void *)result, n * getSizeFromCudaDataType(resultType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDotcEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &yType, sizeof(yType));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_write(client, &resultType, sizeof(resultType));
    rpc_write(client, &executionType, sizeof(executionType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    mem2client((void *)result, n * getSizeFromCudaDataType(resultType));
    return _result;
}

extern "C" cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasSdot_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(float) * n * abs(incy));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSdot_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)y, sizeof(float) * n * abs(incy));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSdot_v2_64(cublasHandle_t handle, int64_t n, const float *x, int64_t incx, const float *y, int64_t incy, float *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasSdot_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(float) * n * abs(incy));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSdot_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)y, sizeof(float) * n * abs(incy));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasDdot_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(double) * n * abs(incy));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDdot_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)y, sizeof(double) * n * abs(incy));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDdot_v2_64(cublasHandle_t handle, int64_t n, const double *x, int64_t incx, const double *y, int64_t incy, double *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasDdot_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(double) * n * abs(incy));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDdot_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)y, sizeof(double) * n * abs(incy));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCdotu_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasCdotu_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuComplex) * n * abs(incy));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCdotu_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuComplex) * n * abs(incy));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCdotu_v2_64(cublasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, const cuComplex *y, int64_t incy, cuComplex *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasCdotu_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuComplex) * n * abs(incy));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCdotu_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuComplex) * n * abs(incy));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCdotc_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasCdotc_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuComplex) * n * abs(incy));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCdotc_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuComplex) * n * abs(incy));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCdotc_v2_64(cublasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, const cuComplex *y, int64_t incy, cuComplex *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasCdotc_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuComplex) * n * abs(incy));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCdotc_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuComplex) * n * abs(incy));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZdotu_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasZdotu_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZdotu_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZdotu_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, const cuDoubleComplex *y, int64_t incy, cuDoubleComplex *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasZdotu_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZdotu_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZdotc_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasZdotc_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZdotc_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZdotc_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, const cuDoubleComplex *y, int64_t incy, cuDoubleComplex *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasZdotc_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZdotc_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasScalEx(cublasHandle_t handle, int n, const void *alpha, cudaDataType alphaType, void *x, cudaDataType xType, int incx, cudaDataType executionType) {
#ifdef DEBUG
    std::cout << "Hook: cublasScalEx called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasScalEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &alphaType, sizeof(alphaType));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &executionType, sizeof(executionType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    return _result;
}

extern "C" cublasStatus_t cublasScalEx_64(cublasHandle_t handle, int64_t n, const void *alpha, cudaDataType alphaType, void *x, cudaDataType xType, int64_t incx, cudaDataType executionType) {
#ifdef DEBUG
    std::cout << "Hook: cublasScalEx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasScalEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &alphaType, sizeof(alphaType));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &executionType, sizeof(executionType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    return _result;
}

extern "C" cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasSscal_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSscal_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    return _result;
}

extern "C" cublasStatus_t cublasSscal_v2_64(cublasHandle_t handle, int64_t n, const float *alpha, float *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasSscal_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSscal_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    return _result;
}

extern "C" cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDscal_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDscal_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    return _result;
}

extern "C" cublasStatus_t cublasDscal_v2_64(cublasHandle_t handle, int64_t n, const double *alpha, double *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDscal_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDscal_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    return _result;
}

extern "C" cublasStatus_t cublasCscal_v2(cublasHandle_t handle, int n, const cuComplex *alpha, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCscal_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCscal_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    return _result;
}

extern "C" cublasStatus_t cublasCscal_v2_64(cublasHandle_t handle, int64_t n, const cuComplex *alpha, cuComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCscal_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCscal_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    return _result;
}

extern "C" cublasStatus_t cublasCsscal_v2(cublasHandle_t handle, int n, const float *alpha, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsscal_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsscal_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    return _result;
}

extern "C" cublasStatus_t cublasCsscal_v2_64(cublasHandle_t handle, int64_t n, const float *alpha, cuComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsscal_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsscal_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    return _result;
}

extern "C" cublasStatus_t cublasZscal_v2(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZscal_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZscal_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    return _result;
}

extern "C" cublasStatus_t cublasZscal_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZscal_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZscal_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    return _result;
}

extern "C" cublasStatus_t cublasZdscal_v2(cublasHandle_t handle, int n, const double *alpha, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZdscal_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZdscal_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    return _result;
}

extern "C" cublasStatus_t cublasZdscal_v2_64(cublasHandle_t handle, int64_t n, const double *alpha, cuDoubleComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZdscal_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZdscal_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    return _result;
}

extern "C" cublasStatus_t cublasAxpyEx(cublasHandle_t handle, int n, const void *alpha, cudaDataType alphaType, const void *x, cudaDataType xType, int incx, void *y, cudaDataType yType, int incy, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasAxpyEx called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0y = mem2server((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasAxpyEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &alphaType, sizeof(alphaType));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &yType, sizeof(yType));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &executiontype, sizeof(executiontype));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    return _result;
}

extern "C" cublasStatus_t cublasAxpyEx_64(cublasHandle_t handle, int64_t n, const void *alpha, cudaDataType alphaType, const void *x, cudaDataType xType, int64_t incx, void *y, cudaDataType yType, int64_t incy, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasAxpyEx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0y = mem2server((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasAxpyEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &alphaType, sizeof(alphaType));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &yType, sizeof(yType));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &executiontype, sizeof(executiontype));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    return _result;
}

extern "C" cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSaxpy_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(float) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSaxpy_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)y, sizeof(float) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasSaxpy_v2_64(cublasHandle_t handle, int64_t n, const float *alpha, const float *x, int64_t incx, float *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSaxpy_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(float) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSaxpy_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)y, sizeof(float) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDaxpy_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(double) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDaxpy_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)y, sizeof(double) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasDaxpy_v2_64(cublasHandle_t handle, int64_t n, const double *alpha, const double *x, int64_t incx, double *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDaxpy_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(double) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDaxpy_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)y, sizeof(double) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasCaxpy_v2(cublasHandle_t handle, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCaxpy_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuComplex) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCaxpy_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuComplex) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasCaxpy_v2_64(cublasHandle_t handle, int64_t n, const cuComplex *alpha, const cuComplex *x, int64_t incx, cuComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCaxpy_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuComplex) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCaxpy_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuComplex) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasZaxpy_v2(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZaxpy_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZaxpy_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasZaxpy_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int64_t incx, cuDoubleComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZaxpy_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZaxpy_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasCopyEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, void *y, cudaDataType yType, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCopyEx called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0y = mem2server((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCopyEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &yType, sizeof(yType));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    return _result;
}

extern "C" cublasStatus_t cublasCopyEx_64(cublasHandle_t handle, int64_t n, const void *x, cudaDataType xType, int64_t incx, void *y, cudaDataType yType, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCopyEx_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0y = mem2server((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCopyEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &yType, sizeof(yType));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    return _result;
}

extern "C" cublasStatus_t cublasScopy_v2(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasScopy_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(float) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasScopy_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)y, sizeof(float) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasScopy_v2_64(cublasHandle_t handle, int64_t n, const float *x, int64_t incx, float *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasScopy_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(float) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasScopy_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)y, sizeof(float) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasDcopy_v2(cublasHandle_t handle, int n, const double *x, int incx, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDcopy_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(double) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDcopy_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)y, sizeof(double) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasDcopy_v2_64(cublasHandle_t handle, int64_t n, const double *x, int64_t incx, double *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDcopy_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(double) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDcopy_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)y, sizeof(double) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasCcopy_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCcopy_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuComplex) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCcopy_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuComplex) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasCcopy_v2_64(cublasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, cuComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCcopy_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuComplex) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCcopy_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuComplex) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasZcopy_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZcopy_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZcopy_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasZcopy_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, cuDoubleComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZcopy_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZcopy_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasSswap_v2(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSswap_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(float) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSswap_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)y, sizeof(float) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasSswap_v2_64(cublasHandle_t handle, int64_t n, float *x, int64_t incx, float *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSswap_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(float) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSswap_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)y, sizeof(float) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasDswap_v2(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDswap_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(double) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDswap_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)y, sizeof(double) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasDswap_v2_64(cublasHandle_t handle, int64_t n, double *x, int64_t incx, double *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDswap_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(double) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDswap_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)y, sizeof(double) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasCswap_v2(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCswap_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuComplex) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCswap_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuComplex) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasCswap_v2_64(cublasHandle_t handle, int64_t n, cuComplex *x, int64_t incx, cuComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCswap_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuComplex) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCswap_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuComplex) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasZswap_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZswap_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZswap_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasZswap_v2_64(cublasHandle_t handle, int64_t n, cuDoubleComplex *x, int64_t incx, cuDoubleComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZswap_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZswap_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    return _result;
}

extern "C" cublasStatus_t cublasSwapEx(cublasHandle_t handle, int n, void *x, cudaDataType xType, int incx, void *y, cudaDataType yType, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSwapEx called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0y = mem2server((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSwapEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &yType, sizeof(yType));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    return _result;
}

extern "C" cublasStatus_t cublasSwapEx_64(cublasHandle_t handle, int64_t n, void *x, cudaDataType xType, int64_t incx, void *y, cudaDataType yType, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSwapEx_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0y = mem2server((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSwapEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &yType, sizeof(yType));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    return _result;
}

extern "C" cublasStatus_t cublasIsamax_v2(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIsamax_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIsamax_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIsamax_v2_64(cublasHandle_t handle, int64_t n, const float *x, int64_t incx, int64_t *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIsamax_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIsamax_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIdamax_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIdamax_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIdamax_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIdamax_v2_64(cublasHandle_t handle, int64_t n, const double *x, int64_t incx, int64_t *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIdamax_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIdamax_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIcamax_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIcamax_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIcamax_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIcamax_v2_64(cublasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, int64_t *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIcamax_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIcamax_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIzamax_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIzamax_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIzamax_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIzamax_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, int64_t *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIzamax_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIzamax_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIamaxEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIamaxEx called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIamaxEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIamaxEx_64(cublasHandle_t handle, int64_t n, const void *x, cudaDataType xType, int64_t incx, int64_t *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIamaxEx_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIamaxEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIsamin_v2(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIsamin_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIsamin_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIsamin_v2_64(cublasHandle_t handle, int64_t n, const float *x, int64_t incx, int64_t *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIsamin_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIsamin_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIdamin_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIdamin_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIdamin_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIdamin_v2_64(cublasHandle_t handle, int64_t n, const double *x, int64_t incx, int64_t *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIdamin_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIdamin_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIcamin_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIcamin_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIcamin_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIcamin_v2_64(cublasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, int64_t *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIcamin_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIcamin_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIzamin_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIzamin_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIzamin_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIzamin_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, int64_t *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIzamin_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, 0);
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIzamin_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, 0);
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIaminEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, int *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIaminEx called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIaminEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasIaminEx_64(cublasHandle_t handle, int64_t n, const void *x, cudaDataType xType, int64_t incx, int64_t *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasIaminEx_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasIaminEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasAsumEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, void *result, cudaDataType resultType, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasAsumEx called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0result = mem2server((void *)result, n * getSizeFromCudaDataType(resultType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasAsumEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_write(client, &resultType, sizeof(resultType));
    rpc_write(client, &executiontype, sizeof(executiontype));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)result, n * getSizeFromCudaDataType(resultType));
    return _result;
}

extern "C" cublasStatus_t cublasAsumEx_64(cublasHandle_t handle, int64_t n, const void *x, cudaDataType xType, int64_t incx, void *result, cudaDataType resultType, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasAsumEx_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0result = mem2server((void *)result, n * getSizeFromCudaDataType(resultType));
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasAsumEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_write(client, &resultType, sizeof(resultType));
    rpc_write(client, &executiontype, sizeof(executiontype));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)result, n * getSizeFromCudaDataType(resultType));
    return _result;
}

extern "C" cublasStatus_t cublasSasum_v2(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasSasum_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSasum_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSasum_v2_64(cublasHandle_t handle, int64_t n, const float *x, int64_t incx, float *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasSasum_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSasum_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDasum_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasDasum_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDasum_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDasum_v2_64(cublasHandle_t handle, int64_t n, const double *x, int64_t incx, double *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasDasum_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDasum_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasScasum_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, float *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasScasum_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasScasum_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasScasum_v2_64(cublasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx, float *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasScasum_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasScasum_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDzasum_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasDzasum_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDzasum_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDzasum_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex *x, int64_t incx, double *result) {
#ifdef DEBUG
    std::cout << "Hook: cublasDzasum_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0result = mem2server((void *)result, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDzasum_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0result, sizeof(_0result));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)result, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSrot_v2(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *c, const float *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasSrot_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(float) * n * abs(incy));
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSrot_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)y, sizeof(float) * n * abs(incy));
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSrot_v2_64(cublasHandle_t handle, int64_t n, float *x, int64_t incx, float *y, int64_t incy, const float *c, const float *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasSrot_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(float) * n * abs(incy));
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSrot_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)y, sizeof(float) * n * abs(incy));
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDrot_v2(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *c, const double *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasDrot_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(double) * n * abs(incy));
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDrot_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)y, sizeof(double) * n * abs(incy));
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDrot_v2_64(cublasHandle_t handle, int64_t n, double *x, int64_t incx, double *y, int64_t incy, const double *c, const double *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasDrot_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(double) * n * abs(incy));
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDrot_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)y, sizeof(double) * n * abs(incy));
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCrot_v2(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const cuComplex *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasCrot_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuComplex) * n * abs(incy));
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCrot_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuComplex) * n * abs(incy));
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCrot_v2_64(cublasHandle_t handle, int64_t n, cuComplex *x, int64_t incx, cuComplex *y, int64_t incy, const float *c, const cuComplex *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasCrot_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuComplex) * n * abs(incy));
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCrot_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuComplex) * n * abs(incy));
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCsrot_v2(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const float *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsrot_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuComplex) * n * abs(incy));
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsrot_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuComplex) * n * abs(incy));
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCsrot_v2_64(cublasHandle_t handle, int64_t n, cuComplex *x, int64_t incx, cuComplex *y, int64_t incy, const float *c, const float *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsrot_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuComplex) * n * abs(incy));
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsrot_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuComplex) * n * abs(incy));
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZrot_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, const double *c, const cuDoubleComplex *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasZrot_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZrot_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZrot_v2_64(cublasHandle_t handle, int64_t n, cuDoubleComplex *x, int64_t incx, cuDoubleComplex *y, int64_t incy, const double *c, const cuDoubleComplex *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasZrot_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZrot_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZdrot_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, const double *c, const double *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasZdrot_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZdrot_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZdrot_v2_64(cublasHandle_t handle, int64_t n, cuDoubleComplex *x, int64_t incx, cuDoubleComplex *y, int64_t incy, const double *c, const double *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasZdrot_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZdrot_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(cuDoubleComplex) * n * abs(incx));
    mem2client((void *)y, sizeof(cuDoubleComplex) * n * abs(incy));
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasRotEx(cublasHandle_t handle, int n, void *x, cudaDataType xType, int incx, void *y, cudaDataType yType, int incy, const void *c, const void *s, cudaDataType csType, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasRotEx called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0y = mem2server((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasRotEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &yType, sizeof(yType));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_write(client, &csType, sizeof(csType));
    rpc_write(client, &executiontype, sizeof(executiontype));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasRotEx_64(cublasHandle_t handle, int64_t n, void *x, cudaDataType xType, int64_t incx, void *y, cudaDataType yType, int64_t incy, const void *c, const void *s, cudaDataType csType, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasRotEx_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0y = mem2server((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasRotEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &yType, sizeof(yType));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_write(client, &csType, sizeof(csType));
    rpc_write(client, &executiontype, sizeof(executiontype));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSrotg_v2(cublasHandle_t handle, float *a, float *b, float *c, float *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasSrotg_v2 called" << std::endl;
#endif
    void *_0a = mem2server((void *)a, 0);
    void *_0b = mem2server((void *)b, 0);
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSrotg_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &_0a, sizeof(_0a));
    rpc_write(client, &_0b, sizeof(_0b));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)a, 0);
    mem2client((void *)b, 0);
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDrotg_v2(cublasHandle_t handle, double *a, double *b, double *c, double *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasDrotg_v2 called" << std::endl;
#endif
    void *_0a = mem2server((void *)a, 0);
    void *_0b = mem2server((void *)b, 0);
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDrotg_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &_0a, sizeof(_0a));
    rpc_write(client, &_0b, sizeof(_0b));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)a, 0);
    mem2client((void *)b, 0);
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCrotg_v2(cublasHandle_t handle, cuComplex *a, cuComplex *b, float *c, cuComplex *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasCrotg_v2 called" << std::endl;
#endif
    void *_0a = mem2server((void *)a, 0);
    void *_0b = mem2server((void *)b, 0);
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCrotg_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &_0a, sizeof(_0a));
    rpc_write(client, &_0b, sizeof(_0b));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)a, 0);
    mem2client((void *)b, 0);
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZrotg_v2(cublasHandle_t handle, cuDoubleComplex *a, cuDoubleComplex *b, double *c, cuDoubleComplex *s) {
#ifdef DEBUG
    std::cout << "Hook: cublasZrotg_v2 called" << std::endl;
#endif
    void *_0a = mem2server((void *)a, 0);
    void *_0b = mem2server((void *)b, 0);
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZrotg_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &_0a, sizeof(_0a));
    rpc_write(client, &_0b, sizeof(_0b));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)a, 0);
    mem2client((void *)b, 0);
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasRotgEx(cublasHandle_t handle, void *a, void *b, cudaDataType abType, void *c, void *s, cudaDataType csType, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasRotgEx called" << std::endl;
#endif
    void *_0a = mem2server((void *)a, 0);
    void *_0b = mem2server((void *)b, 0);
    void *_0c = mem2server((void *)c, 0);
    void *_0s = mem2server((void *)s, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasRotgEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &_0a, sizeof(_0a));
    rpc_write(client, &_0b, sizeof(_0b));
    rpc_write(client, &abType, sizeof(abType));
    rpc_write(client, &_0c, sizeof(_0c));
    rpc_write(client, &_0s, sizeof(_0s));
    rpc_write(client, &csType, sizeof(csType));
    rpc_write(client, &executiontype, sizeof(executiontype));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)a, 0);
    mem2client((void *)b, 0);
    mem2client((void *)c, 0);
    mem2client((void *)s, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSrotm_v2(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *param) {
#ifdef DEBUG
    std::cout << "Hook: cublasSrotm_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(float) * n * abs(incy));
    void *_0param = mem2server((void *)param, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSrotm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0param, sizeof(_0param));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)y, sizeof(float) * n * abs(incy));
    mem2client((void *)param, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSrotm_v2_64(cublasHandle_t handle, int64_t n, float *x, int64_t incx, float *y, int64_t incy, const float *param) {
#ifdef DEBUG
    std::cout << "Hook: cublasSrotm_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(float) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(float) * n * abs(incy));
    void *_0param = mem2server((void *)param, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSrotm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0param, sizeof(_0param));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(float) * n * abs(incx));
    mem2client((void *)y, sizeof(float) * n * abs(incy));
    mem2client((void *)param, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDrotm_v2(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *param) {
#ifdef DEBUG
    std::cout << "Hook: cublasDrotm_v2 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(double) * n * abs(incy));
    void *_0param = mem2server((void *)param, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDrotm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0param, sizeof(_0param));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)y, sizeof(double) * n * abs(incy));
    mem2client((void *)param, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDrotm_v2_64(cublasHandle_t handle, int64_t n, double *x, int64_t incx, double *y, int64_t incy, const double *param) {
#ifdef DEBUG
    std::cout << "Hook: cublasDrotm_v2_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, sizeof(double) * n * abs(incx));
    void *_0y = mem2server((void *)y, sizeof(double) * n * abs(incy));
    void *_0param = mem2server((void *)param, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDrotm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0param, sizeof(_0param));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, sizeof(double) * n * abs(incx));
    mem2client((void *)y, sizeof(double) * n * abs(incy));
    mem2client((void *)param, 0);
    return _result;
}

extern "C" cublasStatus_t cublasRotmEx(cublasHandle_t handle, int n, void *x, cudaDataType xType, int incx, void *y, cudaDataType yType, int incy, const void *param, cudaDataType paramType, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasRotmEx called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0y = mem2server((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    void *_0param = mem2server((void *)param, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasRotmEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &yType, sizeof(yType));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0param, sizeof(_0param));
    rpc_write(client, &paramType, sizeof(paramType));
    rpc_write(client, &executiontype, sizeof(executiontype));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    mem2client((void *)param, 0);
    return _result;
}

extern "C" cublasStatus_t cublasRotmEx_64(cublasHandle_t handle, int64_t n, void *x, cudaDataType xType, int64_t incx, void *y, cudaDataType yType, int64_t incy, const void *param, cudaDataType paramType, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasRotmEx_64 called" << std::endl;
#endif
    void *_0x = mem2server((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    void *_0y = mem2server((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    void *_0param = mem2server((void *)param, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasRotmEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &xType, sizeof(xType));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &yType, sizeof(yType));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0param, sizeof(_0param));
    rpc_write(client, &paramType, sizeof(paramType));
    rpc_write(client, &executiontype, sizeof(executiontype));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)x, (1 + (n - 1) * abs(incx)) * getSizeFromCudaDataType(xType));
    mem2client((void *)y, (1 + (n - 1) * abs(incy)) * getSizeFromCudaDataType(yType));
    mem2client((void *)param, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSrotmg_v2(cublasHandle_t handle, float *d1, float *d2, float *x1, const float *y1, float *param) {
#ifdef DEBUG
    std::cout << "Hook: cublasSrotmg_v2 called" << std::endl;
#endif
    void *_0d1 = mem2server((void *)d1, 0);
    void *_0d2 = mem2server((void *)d2, 0);
    void *_0x1 = mem2server((void *)x1, 0);
    void *_0y1 = mem2server((void *)y1, 0);
    void *_0param = mem2server((void *)param, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSrotmg_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &_0d1, sizeof(_0d1));
    rpc_write(client, &_0d2, sizeof(_0d2));
    rpc_write(client, &_0x1, sizeof(_0x1));
    rpc_write(client, &_0y1, sizeof(_0y1));
    rpc_write(client, &_0param, sizeof(_0param));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)d1, 0);
    mem2client((void *)d2, 0);
    mem2client((void *)x1, 0);
    mem2client((void *)y1, 0);
    mem2client((void *)param, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDrotmg_v2(cublasHandle_t handle, double *d1, double *d2, double *x1, const double *y1, double *param) {
#ifdef DEBUG
    std::cout << "Hook: cublasDrotmg_v2 called" << std::endl;
#endif
    void *_0d1 = mem2server((void *)d1, 0);
    void *_0d2 = mem2server((void *)d2, 0);
    void *_0x1 = mem2server((void *)x1, 0);
    void *_0y1 = mem2server((void *)y1, 0);
    void *_0param = mem2server((void *)param, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDrotmg_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &_0d1, sizeof(_0d1));
    rpc_write(client, &_0d2, sizeof(_0d2));
    rpc_write(client, &_0x1, sizeof(_0x1));
    rpc_write(client, &_0y1, sizeof(_0y1));
    rpc_write(client, &_0param, sizeof(_0param));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)d1, 0);
    mem2client((void *)d2, 0);
    mem2client((void *)x1, 0);
    mem2client((void *)y1, 0);
    mem2client((void *)param, 0);
    return _result;
}

extern "C" cublasStatus_t cublasRotmgEx(cublasHandle_t handle, void *d1, cudaDataType d1Type, void *d2, cudaDataType d2Type, void *x1, cudaDataType x1Type, const void *y1, cudaDataType y1Type, void *param, cudaDataType paramType, cudaDataType executiontype) {
#ifdef DEBUG
    std::cout << "Hook: cublasRotmgEx called" << std::endl;
#endif
    void *_0d1 = mem2server((void *)d1, 0);
    void *_0d2 = mem2server((void *)d2, 0);
    void *_0x1 = mem2server((void *)x1, 0);
    void *_0y1 = mem2server((void *)y1, 0);
    void *_0param = mem2server((void *)param, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasRotmgEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &_0d1, sizeof(_0d1));
    rpc_write(client, &d1Type, sizeof(d1Type));
    rpc_write(client, &_0d2, sizeof(_0d2));
    rpc_write(client, &d2Type, sizeof(d2Type));
    rpc_write(client, &_0x1, sizeof(_0x1));
    rpc_write(client, &x1Type, sizeof(x1Type));
    rpc_write(client, &_0y1, sizeof(_0y1));
    rpc_write(client, &y1Type, sizeof(y1Type));
    rpc_write(client, &_0param, sizeof(_0param));
    rpc_write(client, &paramType, sizeof(paramType));
    rpc_write(client, &executiontype, sizeof(executiontype));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)d1, 0);
    mem2client((void *)d2, 0);
    mem2client((void *)x1, 0);
    mem2client((void *)y1, 0);
    mem2client((void *)param, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float *alpha, const float *A, int64_t lda, const float *x, int64_t incx, const float *beta, float *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgemv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double *alpha, const double *A, int64_t lda, const double *x, int64_t incx, const double *beta, double *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgemv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *x, int64_t incx, const cuComplex *beta, cuComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgemv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *x, int64_t incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgemv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgbmv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgbmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &kl, sizeof(kl));
    rpc_write(client, &ku, sizeof(ku));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgbmv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const float *alpha, const float *A, int64_t lda, const float *x, int64_t incx, const float *beta, float *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgbmv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgbmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &kl, sizeof(kl));
    rpc_write(client, &ku, sizeof(ku));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgbmv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgbmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &kl, sizeof(kl));
    rpc_write(client, &ku, sizeof(ku));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgbmv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const double *alpha, const double *A, int64_t lda, const double *x, int64_t incx, const double *beta, double *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgbmv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgbmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &kl, sizeof(kl));
    rpc_write(client, &ku, sizeof(ku));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgbmv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgbmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &kl, sizeof(kl));
    rpc_write(client, &ku, sizeof(ku));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgbmv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *x, int64_t incx, const cuComplex *beta, cuComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgbmv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgbmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &kl, sizeof(kl));
    rpc_write(client, &ku, sizeof(ku));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgbmv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgbmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &kl, sizeof(kl));
    rpc_write(client, &ku, sizeof(ku));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgbmv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *x, int64_t incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgbmv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgbmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &kl, sizeof(kl));
    rpc_write(client, &ku, sizeof(ku));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *A, int lda, float *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrmv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStrmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float *A, int64_t lda, float *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrmv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStrmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrmv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtrmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double *A, int64_t lda, double *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrmv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtrmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrmv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtrmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex *A, int64_t lda, cuComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrmv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtrmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrmv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtrmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex *A, int64_t lda, cuDoubleComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrmv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtrmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStbmv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStbmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const float *A, int64_t lda, float *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStbmv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStbmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtbmv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtbmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const double *A, int64_t lda, double *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtbmv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtbmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtbmv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtbmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuComplex *A, int64_t lda, cuComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtbmv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtbmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtbmv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtbmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuDoubleComplex *A, int64_t lda, cuDoubleComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtbmv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtbmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *AP, float *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStpmv_v2 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStpmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float *AP, float *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStpmv_v2_64 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStpmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *AP, double *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtpmv_v2 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtpmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double *AP, double *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtpmv_v2_64 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtpmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtpmv_v2 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtpmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex *AP, cuComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtpmv_v2_64 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtpmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtpmv_v2 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtpmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex *AP, cuDoubleComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtpmv_v2_64 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtpmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *A, int lda, float *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrsv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStrsv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float *A, int64_t lda, float *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrsv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStrsv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrsv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtrsv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double *A, int64_t lda, double *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrsv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtrsv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrsv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtrsv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex *A, int64_t lda, cuComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrsv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtrsv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrsv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtrsv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex *A, int64_t lda, cuDoubleComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrsv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtrsv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *AP, float *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStpsv_v2 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStpsv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float *AP, float *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStpsv_v2_64 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStpsv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *AP, double *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtpsv_v2 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtpsv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double *AP, double *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtpsv_v2_64 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtpsv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtpsv_v2 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtpsv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex *AP, cuComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtpsv_v2_64 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtpsv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtpsv_v2 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtpsv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex *AP, cuDoubleComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtpsv_v2_64 called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtpsv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStbsv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStbsv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const float *A, int64_t lda, float *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasStbsv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStbsv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtbsv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtbsv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const double *A, int64_t lda, double *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtbsv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtbsv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtbsv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtbsv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuComplex *A, int64_t lda, cuComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtbsv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtbsv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtbsv_v2 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtbsv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuDoubleComplex *A, int64_t lda, cuDoubleComplex *x, int64_t incx) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtbsv_v2_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtbsv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)x, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsymv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsymv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float *alpha, const float *A, int64_t lda, const float *x, int64_t incx, const float *beta, float *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsymv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsymv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsymv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsymv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double *alpha, const double *A, int64_t lda, const double *x, int64_t incx, const double *beta, double *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsymv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsymv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsymv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsymv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *x, int64_t incx, const cuComplex *beta, cuComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsymv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsymv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsymv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZsymv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *x, int64_t incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsymv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZsymv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasChemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasChemv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasChemv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasChemv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *x, int64_t incx, const cuComplex *beta, cuComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasChemv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasChemv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZhemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhemv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZhemv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZhemv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *x, int64_t incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhemv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZhemv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsbmv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsbmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSsbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const float *alpha, const float *A, int64_t lda, const float *x, int64_t incx, const float *beta, float *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsbmv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsbmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsbmv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsbmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDsbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const double *alpha, const double *A, int64_t lda, const double *x, int64_t incx, const double *beta, double *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsbmv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsbmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasChbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasChbmv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasChbmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasChbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *x, int64_t incx, const cuComplex *beta, cuComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasChbmv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasChbmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZhbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhbmv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZhbmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZhbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *x, int64_t incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhbmv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZhbmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *AP, const float *x, int incx, const float *beta, float *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSspmv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSspmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSspmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float *alpha, const float *AP, const float *x, int64_t incx, const float *beta, float *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasSspmv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSspmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *AP, const double *x, int incx, const double *beta, double *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDspmv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDspmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDspmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double *alpha, const double *AP, const double *x, int64_t incx, const double *beta, double *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasDspmv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDspmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasChpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *AP, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasChpmv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasChpmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasChpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex *alpha, const cuComplex *AP, const cuComplex *x, int64_t incx, const cuComplex *beta, cuComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasChpmv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasChpmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZhpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *AP, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhpmv_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZhpmv_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZhpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *AP, const cuDoubleComplex *x, int64_t incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int64_t incy) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhpmv_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0AP = mem2server((void *)AP, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZhpmv_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)AP, 0);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSger_v2(cublasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasSger_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSger_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSger_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const float *alpha, const float *x, int64_t incx, const float *y, int64_t incy, float *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasSger_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSger_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDger_v2(cublasHandle_t handle, int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasDger_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDger_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDger_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const double *alpha, const double *x, int64_t incx, const double *y, int64_t incy, double *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasDger_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDger_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgeru_v2(cublasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgeru_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgeru_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgeru_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const cuComplex *alpha, const cuComplex *x, int64_t incx, const cuComplex *y, int64_t incy, cuComplex *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgeru_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgeru_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgerc_v2(cublasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgerc_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgerc_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgerc_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const cuComplex *alpha, const cuComplex *x, int64_t incx, const cuComplex *y, int64_t incy, cuComplex *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgerc_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgerc_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgeru_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgeru_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgeru_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgeru_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int64_t incx, const cuDoubleComplex *y, int64_t incy, cuDoubleComplex *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgeru_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgeru_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgerc_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgerc_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgerc_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgerc_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int64_t incx, const cuDoubleComplex *y, int64_t incy, cuDoubleComplex *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgerc_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgerc_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsyr_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsyr_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float *alpha, const float *x, int64_t incx, float *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsyr_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsyr_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsyr_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsyr_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double *alpha, const double *x, int64_t incx, double *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsyr_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsyr_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyr_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsyr_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex *alpha, const cuComplex *x, int64_t incx, cuComplex *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyr_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsyr_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsyr_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZsyr_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int64_t incx, cuDoubleComplex *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsyr_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZsyr_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const cuComplex *x, int incx, cuComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCher_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCher_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCher_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float *alpha, const cuComplex *x, int64_t incx, cuComplex *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCher_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCher_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZher_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZher_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZher_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double *alpha, const cuDoubleComplex *x, int64_t incx, cuDoubleComplex *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZher_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZher_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasSspr_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSspr_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSspr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float *alpha, const float *x, int64_t incx, float *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasSspr_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSspr_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasDspr_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDspr_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDspr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double *alpha, const double *x, int64_t incx, double *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasDspr_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDspr_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasChpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const cuComplex *x, int incx, cuComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasChpr_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasChpr_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasChpr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float *alpha, const cuComplex *x, int64_t incx, cuComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasChpr_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasChpr_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZhpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhpr_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZhpr_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZhpr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double *alpha, const cuDoubleComplex *x, int64_t incx, cuDoubleComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhpr_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZhpr_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsyr2_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsyr2_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float *alpha, const float *x, int64_t incx, const float *y, int64_t incy, float *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsyr2_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsyr2_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsyr2_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsyr2_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double *alpha, const double *x, int64_t incx, const double *y, int64_t incy, double *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsyr2_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsyr2_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyr2_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsyr2_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex *alpha, const cuComplex *x, int64_t incx, const cuComplex *y, int64_t incy, cuComplex *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyr2_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsyr2_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsyr2_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZsyr2_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int64_t incx, const cuDoubleComplex *y, int64_t incy, cuDoubleComplex *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsyr2_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZsyr2_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCher2_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCher2_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCher2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex *alpha, const cuComplex *x, int64_t incx, const cuComplex *y, int64_t incy, cuComplex *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCher2_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCher2_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZher2_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZher2_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZher2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int64_t incx, const cuDoubleComplex *y, int64_t incy, cuDoubleComplex *A, int64_t lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZher2_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZher2_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasSspr2_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSspr2_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSspr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float *alpha, const float *x, int64_t incx, const float *y, int64_t incy, float *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasSspr2_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSspr2_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasDspr2_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDspr2_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDspr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double *alpha, const double *x, int64_t incx, const double *y, int64_t incy, double *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasDspr2_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDspr2_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasChpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasChpr2_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasChpr2_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasChpr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex *alpha, const cuComplex *x, int64_t incx, const cuComplex *y, int64_t incy, cuComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasChpr2_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasChpr2_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZhpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhpr2_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZhpr2_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZhpr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int64_t incx, const cuDoubleComplex *y, int64_t incy, cuDoubleComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhpr2_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0y = mem2server((void *)y, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZhpr2_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)x, 0);
    mem2client((void *)y, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float* const Aarray[], int lda, const float* const xarray[], int incx, const float *beta, float* const yarray[], int incy, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemvBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemvBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(float *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(float *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(float *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float *alpha, const float* const Aarray[], int64_t lda, const float* const xarray[], int64_t incx, const float *beta, float* const yarray[], int64_t incy, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemvBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemvBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(float *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(float *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(float *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double* const Aarray[], int lda, const double* const xarray[], int incx, const double *beta, double* const yarray[], int incy, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemvBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgemvBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(double *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(double *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(double *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double *alpha, const double* const Aarray[], int64_t lda, const double* const xarray[], int64_t incx, const double *beta, double* const yarray[], int64_t incy, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemvBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgemvBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(double *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(double *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(double *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex *alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const xarray[], int incx, const cuComplex *beta, cuComplex* const yarray[], int incy, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemvBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemvBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex *alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const xarray[], int64_t incx, const cuComplex *beta, cuComplex* const yarray[], int64_t incy, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemvBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemvBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const xarray[], int incx, const cuDoubleComplex *beta, cuDoubleComplex* const yarray[], int incy, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemvBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgemvBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex* const Aarray[], int64_t lda, const cuDoubleComplex* const xarray[], int64_t incx, const cuDoubleComplex *beta, cuDoubleComplex* const yarray[], int64_t incy, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemvBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgemvBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasHSHgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const __half* const Aarray[], int lda, const __half* const xarray[], int incx, const float *beta, __half* const yarray[], int incy, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasHSHgemvBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasHSHgemvBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasHSHgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float *alpha, const __half* const Aarray[], int64_t lda, const __half* const xarray[], int64_t incx, const float *beta, __half* const yarray[], int64_t incy, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasHSHgemvBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasHSHgemvBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasHSSgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const __half* const Aarray[], int lda, const __half* const xarray[], int incx, const float *beta, float* const yarray[], int incy, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasHSSgemvBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasHSSgemvBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(float *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasHSSgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float *alpha, const __half* const Aarray[], int64_t lda, const __half* const xarray[], int64_t incx, const float *beta, float* const yarray[], int64_t incy, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasHSSgemvBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasHSSgemvBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(float *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasTSTgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const __nv_bfloat16* const Aarray[], int lda, const __nv_bfloat16* const xarray[], int incx, const float *beta, __nv_bfloat16* const yarray[], int incy, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasTSTgemvBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasTSTgemvBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(__nv_bfloat16 *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(__nv_bfloat16 *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(__nv_bfloat16 *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasTSTgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float *alpha, const __nv_bfloat16* const Aarray[], int64_t lda, const __nv_bfloat16* const xarray[], int64_t incx, const float *beta, __nv_bfloat16* const yarray[], int64_t incy, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasTSTgemvBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasTSTgemvBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(__nv_bfloat16 *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(__nv_bfloat16 *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(__nv_bfloat16 *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasTSSgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const __nv_bfloat16* const Aarray[], int lda, const __nv_bfloat16* const xarray[], int incx, const float *beta, float* const yarray[], int incy, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasTSSgemvBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasTSSgemvBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(__nv_bfloat16 *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(__nv_bfloat16 *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(float *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasTSSgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float *alpha, const __nv_bfloat16* const Aarray[], int64_t lda, const __nv_bfloat16* const xarray[], int64_t incx, const float *beta, float* const yarray[], int64_t incy, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasTSSgemvBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasTSSgemvBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(__nv_bfloat16 *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, xarray, sizeof(__nv_bfloat16 *)*batchCount, true);
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, yarray, sizeof(float *)*batchCount, true);
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, long long int strideA, const float *x, int incx, long long int stridex, const float *beta, float *y, int incy, long long int stridey, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemvStridedBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemvStridedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float *alpha, const float *A, int64_t lda, long long int strideA, const float *x, int64_t incx, long long int stridex, const float *beta, float *y, int64_t incy, long long int stridey, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemvStridedBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemvStridedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, long long int strideA, const double *x, int incx, long long int stridex, const double *beta, double *y, int incy, long long int stridey, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemvStridedBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgemvStridedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double *alpha, const double *A, int64_t lda, long long int strideA, const double *x, int64_t incx, long long int stridex, const double *beta, double *y, int64_t incy, long long int stridey, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemvStridedBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgemvStridedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, long long int strideA, const cuComplex *x, int incx, long long int stridex, const cuComplex *beta, cuComplex *y, int incy, long long int stridey, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemvStridedBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemvStridedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex *alpha, const cuComplex *A, int64_t lda, long long int strideA, const cuComplex *x, int64_t incx, long long int stridex, const cuComplex *beta, cuComplex *y, int64_t incy, long long int stridey, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemvStridedBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemvStridedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, long long int strideA, const cuDoubleComplex *x, int incx, long long int stridex, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy, long long int stridey, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemvStridedBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgemvStridedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, long long int strideA, const cuDoubleComplex *x, int64_t incx, long long int stridex, const cuDoubleComplex *beta, cuDoubleComplex *y, int64_t incy, long long int stridey, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemvStridedBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgemvStridedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasHSHgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const __half *A, int lda, long long int strideA, const __half *x, int incx, long long int stridex, const float *beta, __half *y, int incy, long long int stridey, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasHSHgemvStridedBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(__half) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasHSHgemvStridedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(__half) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasHSHgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float *alpha, const __half *A, int64_t lda, long long int strideA, const __half *x, int64_t incx, long long int stridex, const float *beta, __half *y, int64_t incy, long long int stridey, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasHSHgemvStridedBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(__half) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasHSHgemvStridedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(__half) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasHSSgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const __half *A, int lda, long long int strideA, const __half *x, int incx, long long int stridex, const float *beta, float *y, int incy, long long int stridey, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasHSSgemvStridedBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(__half) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasHSSgemvStridedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(__half) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasHSSgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float *alpha, const __half *A, int64_t lda, long long int strideA, const __half *x, int64_t incx, long long int stridex, const float *beta, float *y, int64_t incy, long long int stridey, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasHSSgemvStridedBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(__half) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasHSSgemvStridedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(__half) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasTSTgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const __nv_bfloat16 *A, int lda, long long int strideA, const __nv_bfloat16 *x, int incx, long long int stridex, const float *beta, __nv_bfloat16 *y, int incy, long long int stridey, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasTSTgemvStridedBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(__nv_bfloat16) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasTSTgemvStridedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(__nv_bfloat16) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasTSTgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float *alpha, const __nv_bfloat16 *A, int64_t lda, long long int strideA, const __nv_bfloat16 *x, int64_t incx, long long int stridex, const float *beta, __nv_bfloat16 *y, int64_t incy, long long int stridey, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasTSTgemvStridedBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(__nv_bfloat16) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasTSTgemvStridedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(__nv_bfloat16) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasTSSgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const __nv_bfloat16 *A, int lda, long long int strideA, const __nv_bfloat16 *x, int incx, long long int stridex, const float *beta, float *y, int incy, long long int stridey, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasTSSgemvStridedBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(__nv_bfloat16) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasTSSgemvStridedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(__nv_bfloat16) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasTSSgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float *alpha, const __nv_bfloat16 *A, int64_t lda, long long int strideA, const __nv_bfloat16 *x, int64_t incx, long long int stridex, const float *beta, float *y, int64_t incy, long long int stridey, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasTSSgemvStridedBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(__nv_bfloat16) * lda * n);
    void *_0x = mem2server((void *)x, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0y = mem2server((void *)y, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasTSSgemvStridedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &stridex, sizeof(stridex));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0y, sizeof(_0y));
    rpc_write(client, &incy, sizeof(incy));
    rpc_write(client, &stridey, sizeof(stridey));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(__nv_bfloat16) * lda * n);
    mem2client((void *)x, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)y, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(float) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(float) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(float) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(float) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasSgemm_v2_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float *alpha, const float *A, int64_t lda, const float *B, int64_t ldb, const float *beta, float *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(float) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(float) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(float) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(float) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(double) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(double) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgemm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(double) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(double) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasDgemm_v2_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double *alpha, const double *A, int64_t lda, const double *B, int64_t ldb, const double *beta, double *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(double) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(double) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgemm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(double) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(double) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCgemm_v2_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *B, int64_t ldb, const cuComplex *beta, cuComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemm3m called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemm3m);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCgemm3m_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *B, int64_t ldb, const cuComplex *beta, cuComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemm3m_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemm3m_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCgemm3mEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const void *B, cudaDataType Btype, int ldb, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemm3mEx called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(void) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(void) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(void) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemm3mEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &Btype, sizeof(Btype));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(void) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(void) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(void) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCgemm3mEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex *alpha, const void *A, cudaDataType Atype, int64_t lda, const void *B, cudaDataType Btype, int64_t ldb, const cuComplex *beta, void *C, cudaDataType Ctype, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemm3mEx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(void) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(void) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(void) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemm3mEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &Btype, sizeof(Btype));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(void) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(void) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(void) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgemm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZgemm_v2_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *B, int64_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgemm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemm3m called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgemm3m);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZgemm3m_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *B, int64_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemm3m_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgemm3m_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, const __half *B, int ldb, const __half *beta, __half *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasHgemm called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(__half) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(__half) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(__half) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasHgemm);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(__half) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(__half) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(__half) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasHgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const __half *alpha, const __half *A, int64_t lda, const __half *B, int64_t ldb, const __half *beta, __half *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasHgemm_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(__half) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(__half) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(__half) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasHgemm_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(__half) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(__half) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(__half) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const void *A, cudaDataType Atype, int lda, const void *B, cudaDataType Btype, int ldb, const float *beta, void *C, cudaDataType Ctype, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemmEx called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(void) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(void) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(void) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemmEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &Btype, sizeof(Btype));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(void) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(void) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(void) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasSgemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float *alpha, const void *A, cudaDataType Atype, int64_t lda, const void *B, cudaDataType Btype, int64_t ldb, const float *beta, void *C, cudaDataType Ctype, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemmEx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(void) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(void) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(void) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemmEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &Btype, sizeof(Btype));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(void) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(void) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(void) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasGemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void *alpha, const void *A, cudaDataType Atype, int64_t lda, const void *B, cudaDataType Btype, int64_t ldb, const void *beta, void *C, cudaDataType Ctype, int64_t ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
#ifdef DEBUG
    std::cout << "Hook: cublasGemmEx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGemmEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &Btype, sizeof(Btype));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &computeType, sizeof(computeType));
    rpc_write(client, &algo, sizeof(algo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const void *B, cudaDataType Btype, int ldb, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemmEx called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(void) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(void) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(void) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemmEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &Btype, sizeof(Btype));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(void) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(void) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(void) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCgemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex *alpha, const void *A, cudaDataType Atype, int64_t lda, const void *B, cudaDataType Btype, int64_t ldb, const cuComplex *beta, void *C, cudaDataType Ctype, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemmEx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(void) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(void) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(void) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemmEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &Btype, sizeof(Btype));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(void) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(void) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(void) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasSsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *beta, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsyrk_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(float) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsyrk_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(float) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasSsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float *alpha, const float *A, int64_t lda, const float *beta, float *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsyrk_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(float) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsyrk_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(float) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *beta, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsyrk_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(double) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsyrk_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(double) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasDsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double *alpha, const double *A, int64_t lda, const double *beta, double *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsyrk_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(double) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsyrk_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(double) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyrk_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsyrk_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *beta, cuComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyrk_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsyrk_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsyrk_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZsyrk_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *beta, cuDoubleComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsyrk_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZsyrk_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyrkEx called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(void) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(void) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsyrkEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(void) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(void) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCsyrkEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex *alpha, const void *A, cudaDataType Atype, int64_t lda, const cuComplex *beta, void *C, cudaDataType Ctype, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyrkEx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(void) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(void) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsyrkEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(void) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(void) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCsyrk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyrk3mEx called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(void) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(void) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsyrk3mEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(void) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(void) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCsyrk3mEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex *alpha, const void *A, cudaDataType Atype, int64_t lda, const cuComplex *beta, void *C, cudaDataType Ctype, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyrk3mEx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(void) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(void) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsyrk3mEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(void) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(void) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const cuComplex *A, int lda, const float *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCherk_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCherk_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCherk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float *alpha, const cuComplex *A, int64_t lda, const float *beta, cuComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCherk_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCherk_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const cuDoubleComplex *A, int lda, const double *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZherk_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZherk_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZherk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double *alpha, const cuDoubleComplex *A, int64_t lda, const double *beta, cuDoubleComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZherk_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZherk_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCherkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const void *A, cudaDataType Atype, int lda, const float *beta, void *C, cudaDataType Ctype, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCherkEx called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(void) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(void) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCherkEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(void) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(void) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCherkEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float *alpha, const void *A, cudaDataType Atype, int64_t lda, const float *beta, void *C, cudaDataType Ctype, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCherkEx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(void) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(void) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCherkEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(void) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(void) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCherk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const void *A, cudaDataType Atype, int lda, const float *beta, void *C, cudaDataType Ctype, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCherk3mEx called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(void) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(void) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCherk3mEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(void) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(void) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCherk3mEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float *alpha, const void *A, cudaDataType Atype, int64_t lda, const float *beta, void *C, cudaDataType Ctype, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCherk3mEx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(void) * lda * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(void) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCherk3mEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(void) * lda * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(void) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasSsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsyr2k_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(float) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(float) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsyr2k_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)B, sizeof(float) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(float) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasSsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float *alpha, const float *A, int64_t lda, const float *B, int64_t ldb, const float *beta, float *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsyr2k_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(float) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(float) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsyr2k_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)B, sizeof(float) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(float) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasDsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsyr2k_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(double) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(double) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsyr2k_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)B, sizeof(double) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(double) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasDsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double *alpha, const double *A, int64_t lda, const double *B, int64_t ldb, const double *beta, double *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsyr2k_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(double) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(double) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsyr2k_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)B, sizeof(double) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(double) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyr2k_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsyr2k_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)B, sizeof(cuComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *B, int64_t ldb, const cuComplex *beta, cuComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyr2k_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsyr2k_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)B, sizeof(cuComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsyr2k_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZsyr2k_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *B, int64_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsyr2k_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZsyr2k_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCher2k_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCher2k_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)B, sizeof(cuComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCher2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *B, int64_t ldb, const float *beta, cuComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCher2k_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCher2k_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)B, sizeof(cuComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZher2k_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZher2k_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZher2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *B, int64_t ldb, const double *beta, cuDoubleComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZher2k_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZher2k_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsyrkx called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(float) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(float) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsyrkx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)B, sizeof(float) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(float) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasSsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float *alpha, const float *A, int64_t lda, const float *B, int64_t ldb, const float *beta, float *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsyrkx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(float) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(float) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsyrkx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)B, sizeof(float) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(float) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsyrkx called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(double) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(double) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsyrkx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)B, sizeof(double) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(double) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasDsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double *alpha, const double *A, int64_t lda, const double *B, int64_t ldb, const double *beta, double *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsyrkx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(double) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(double) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsyrkx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)B, sizeof(double) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(double) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyrkx called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsyrkx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)B, sizeof(cuComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *B, int64_t ldb, const cuComplex *beta, cuComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsyrkx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsyrkx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)B, sizeof(cuComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsyrkx called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZsyrkx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *B, int64_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsyrkx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZsyrkx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCherkx called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCherkx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)B, sizeof(cuComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCherkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *B, int64_t ldb, const float *beta, cuComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCherkx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCherkx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)B, sizeof(cuComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZherkx called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZherkx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZherkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *B, int64_t ldb, const double *beta, cuDoubleComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZherkx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZherkx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasSsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsymm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(float) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(float) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsymm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)B, sizeof(float) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(float) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasSsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const float *alpha, const float *A, int64_t lda, const float *B, int64_t ldb, const float *beta, float *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSsymm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(float) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(float) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSsymm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * n);
    mem2client((void *)B, sizeof(float) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(float) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasDsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsymm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(double) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(double) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsymm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)B, sizeof(double) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(double) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasDsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const double *alpha, const double *A, int64_t lda, const double *B, int64_t ldb, const double *beta, double *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDsymm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(double) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(double) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDsymm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * n);
    mem2client((void *)B, sizeof(double) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(double) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsymm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsymm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)B, sizeof(cuComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *B, int64_t ldb, const cuComplex *beta, cuComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCsymm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCsymm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)B, sizeof(cuComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsymm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZsymm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *B, int64_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZsymm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZsymm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasChemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasChemm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasChemm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)B, sizeof(cuComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasChemm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *B, int64_t ldb, const cuComplex *beta, cuComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasChemm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasChemm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * n);
    mem2client((void *)B, sizeof(cuComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZhemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhemm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZhemm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZhemm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *B, int64_t ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZhemm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * n);
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZhemm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * n);
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * n);
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrsm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStrsm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float *alpha, const float *A, int64_t lda, float *B, int64_t ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrsm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStrsm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrsm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtrsm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double *alpha, const double *A, int64_t lda, double *B, int64_t ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrsm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtrsm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, cuComplex *B, int ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrsm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtrsm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex *alpha, const cuComplex *A, int64_t lda, cuComplex *B, int64_t ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrsm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtrsm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrsm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtrsm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, cuDoubleComplex *B, int64_t ldb) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrsm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtrsm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrmm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStrmm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float *alpha, const float *A, int64_t lda, const float *B, int64_t ldb, float *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrmm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStrmm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrmm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtrmm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double *alpha, const double *A, int64_t lda, const double *B, int64_t ldb, double *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrmm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtrmm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrmm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtrmm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *B, int64_t ldb, cuComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrmm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtrmm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrmm_v2 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtrmm_v2);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *B, int64_t ldb, cuDoubleComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrmm_v2_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtrmm_v2_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasHgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half* const Aarray[], int lda, const __half* const Barray[], int ldb, const __half *beta, __half* const Carray[], int ldc, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasHgemmBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasHgemmBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Barray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, Carray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasHgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const __half *alpha, const __half* const Aarray[], int64_t lda, const __half* const Barray[], int64_t ldb, const __half *beta, __half* const Carray[], int64_t ldc, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasHgemmBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasHgemmBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Barray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, Carray, sizeof(__half *)*batchCount, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float* const Aarray[], int lda, const float* const Barray[], int ldb, const float *beta, float* const Carray[], int ldc, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemmBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemmBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(float *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Barray, sizeof(float *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, Carray, sizeof(float *)*batchCount, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float *alpha, const float* const Aarray[], int64_t lda, const float* const Barray[], int64_t ldb, const float *beta, float* const Carray[], int64_t ldc, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemmBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemmBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(float *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Barray, sizeof(float *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, Carray, sizeof(float *)*batchCount, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double* const Aarray[], int lda, const double* const Barray[], int ldb, const double *beta, double* const Carray[], int ldc, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemmBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgemmBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(double *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Barray, sizeof(double *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, Carray, sizeof(double *)*batchCount, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double *alpha, const double* const Aarray[], int64_t lda, const double* const Barray[], int64_t ldb, const double *beta, double* const Carray[], int64_t ldc, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemmBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgemmBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(double *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Barray, sizeof(double *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, Carray, sizeof(double *)*batchCount, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex *beta, cuComplex* const Carray[], int ldc, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemmBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemmBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Barray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, Carray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex *alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const Barray[], int64_t ldb, const cuComplex *beta, cuComplex* const Carray[], int64_t ldc, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemmBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemmBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Barray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, Carray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgemm3mBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex *beta, cuComplex* const Carray[], int ldc, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemm3mBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemm3mBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Barray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, Carray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgemm3mBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex *alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const Barray[], int64_t ldb, const cuComplex *beta, cuComplex* const Carray[], int64_t ldc, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemm3mBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemm3mBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Barray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, Carray, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const Barray[], int ldb, const cuDoubleComplex *beta, cuDoubleComplex* const Carray[], int ldc, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemmBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgemmBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Barray, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, Carray, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex *alpha, const cuDoubleComplex* const Aarray[], int64_t lda, const cuDoubleComplex* const Barray[], int64_t ldb, const cuDoubleComplex *beta, cuDoubleComplex* const Carray[], int64_t ldc, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemmBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgemmBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Barray, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, Carray, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, long long int strideA, const __half *B, int ldb, long long int strideB, const __half *beta, __half *C, int ldc, long long int strideC, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasHgemmStridedBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(__half) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(__half) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(__half) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasHgemmStridedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &strideB, sizeof(strideB));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &strideC, sizeof(strideC));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(__half) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(__half) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(__half) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasHgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const __half *alpha, const __half *A, int64_t lda, long long int strideA, const __half *B, int64_t ldb, long long int strideB, const __half *beta, __half *C, int64_t ldc, long long int strideC, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasHgemmStridedBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(__half) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(__half) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(__half) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasHgemmStridedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &strideB, sizeof(strideB));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &strideC, sizeof(strideC));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(__half) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(__half) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(__half) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemmStridedBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(float) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(float) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemmStridedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &strideB, sizeof(strideB));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &strideC, sizeof(strideC));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(float) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(float) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasSgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float *alpha, const float *A, int64_t lda, long long int strideA, const float *B, int64_t ldb, long long int strideB, const float *beta, float *C, int64_t ldc, long long int strideC, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemmStridedBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(float) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(float) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(float) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemmStridedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &strideB, sizeof(strideB));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &strideC, sizeof(strideC));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(float) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(float) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(float) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, long long int strideA, const double *B, int ldb, long long int strideB, const double *beta, double *C, int ldc, long long int strideC, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemmStridedBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(double) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(double) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgemmStridedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &strideB, sizeof(strideB));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &strideC, sizeof(strideC));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(double) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(double) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasDgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double *alpha, const double *A, int64_t lda, long long int strideA, const double *B, int64_t ldb, long long int strideB, const double *beta, double *C, int64_t ldc, long long int strideC, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemmStridedBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(double) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(double) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(double) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgemmStridedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &strideB, sizeof(strideB));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &strideC, sizeof(strideC));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(double) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(double) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(double) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, long long int strideA, const cuComplex *B, int ldb, long long int strideB, const cuComplex *beta, cuComplex *C, int ldc, long long int strideC, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemmStridedBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemmStridedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &strideB, sizeof(strideB));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &strideC, sizeof(strideC));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex *alpha, const cuComplex *A, int64_t lda, long long int strideA, const cuComplex *B, int64_t ldb, long long int strideB, const cuComplex *beta, cuComplex *C, int64_t ldc, long long int strideC, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemmStridedBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemmStridedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &strideB, sizeof(strideB));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &strideC, sizeof(strideC));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCgemm3mStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, long long int strideA, const cuComplex *B, int ldb, long long int strideB, const cuComplex *beta, cuComplex *C, int ldc, long long int strideC, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemm3mStridedBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemm3mStridedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &strideB, sizeof(strideB));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &strideC, sizeof(strideC));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasCgemm3mStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex *alpha, const cuComplex *A, int64_t lda, long long int strideA, const cuComplex *B, int64_t ldb, long long int strideB, const cuComplex *beta, cuComplex *C, int64_t ldc, long long int strideC, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgemm3mStridedBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgemm3mStridedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &strideB, sizeof(strideB));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &strideC, sizeof(strideC));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(cuComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, long long int strideA, const cuDoubleComplex *B, int ldb, long long int strideB, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc, long long int strideC, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemmStridedBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgemmStridedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &strideB, sizeof(strideB));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &strideC, sizeof(strideC));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasZgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, long long int strideA, const cuDoubleComplex *B, int64_t ldb, long long int strideB, const cuDoubleComplex *beta, cuDoubleComplex *C, int64_t ldc, long long int strideC, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgemmStridedBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, sizeof(cuDoubleComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(cuDoubleComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgemmStridedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &strideB, sizeof(strideB));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &strideC, sizeof(strideC));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, sizeof(cuDoubleComplex) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(cuDoubleComplex) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)beta, 0);
    mem2client((void *)C, sizeof(cuDoubleComplex) * ldc * n);
    return _result;
}

extern "C" cublasStatus_t cublasGemmBatchedEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void *alpha, const void* const Aarray[], cudaDataType Atype, int64_t lda, const void* const Barray[], cudaDataType Btype, int64_t ldb, const void *beta, void* const Carray[], cudaDataType Ctype, int64_t ldc, int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
#ifdef DEBUG
    std::cout << "Hook: cublasGemmBatchedEx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0beta = mem2server((void *)beta, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGemmBatchedEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, Aarray, sizeof(void *)*batchCount, true);
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Barray, sizeof(void *)*batchCount, true);
    rpc_write(client, &Btype, sizeof(Btype));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, Carray, sizeof(void *)*batchCount, true);
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_write(client, &computeType, sizeof(computeType));
    rpc_write(client, &algo, sizeof(algo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)beta, 0);
    return _result;
}

extern "C" cublasStatus_t cublasGemmStridedBatchedEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void *alpha, const void *A, cudaDataType Atype, int64_t lda, long long int strideA, const void *B, cudaDataType Btype, int64_t ldb, long long int strideB, const void *beta, void *C, cudaDataType Ctype, int64_t ldc, long long int strideC, int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
#ifdef DEBUG
    std::cout << "Hook: cublasGemmStridedBatchedEx_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGemmStridedBatchedEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &strideA, sizeof(strideA));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &Btype, sizeof(Btype));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &strideB, sizeof(strideB));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &strideC, sizeof(strideC));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_write(client, &computeType, sizeof(computeType));
    rpc_write(client, &algo, sizeof(algo));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)B, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgemmGroupedBatched(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int m_array[], const int n_array[], const int k_array[], const float alpha_array[], const float* const Aarray[], const int lda_array[], const float* const Barray[], const int ldb_array[], const float beta_array[], float* const Carray[], const int ldc_array[], int group_count, const int group_size[]) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemmGroupedBatched called" << std::endl;
#endif
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemmGroupedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, transa_array, sizeof(cublasOperation_t *)*group_count, true);
    rpc_write(client, transb_array, sizeof(cublasOperation_t *)*group_count, true);
    rpc_write(client, m_array, sizeof(int *)*group_count, true);
    rpc_write(client, n_array, sizeof(int *)*group_count, true);
    rpc_write(client, k_array, sizeof(int *)*group_count, true);
    rpc_write(client, alpha_array, sizeof(float *)*group_count, true);
    rpc_write(client, Aarray, sizeof(float *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, lda_array, sizeof(int *)*group_count, true);
    rpc_write(client, Barray, sizeof(float *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, ldb_array, sizeof(int *)*group_count, true);
    rpc_write(client, beta_array, sizeof(float *)*group_count, true);
    rpc_write(client, Carray, sizeof(float *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, ldc_array, sizeof(int *)*group_count, true);
    rpc_write(client, &group_count, sizeof(group_count));
    rpc_write(client, group_size, sizeof(int *)*group_count, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cublasStatus_t cublasSgemmGroupedBatched_64(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int64_t m_array[], const int64_t n_array[], const int64_t k_array[], const float alpha_array[], const float* const Aarray[], const int64_t lda_array[], const float* const Barray[], const int64_t ldb_array[], const float beta_array[], float* const Carray[], const int64_t ldc_array[], int64_t group_count, const int64_t group_size[]) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgemmGroupedBatched_64 called" << std::endl;
#endif
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgemmGroupedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, transa_array, sizeof(cublasOperation_t *)*group_count, true);
    rpc_write(client, transb_array, sizeof(cublasOperation_t *)*group_count, true);
    rpc_write(client, m_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, n_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, k_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, alpha_array, sizeof(float *)*group_count, true);
    rpc_write(client, Aarray, sizeof(float *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, lda_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, Barray, sizeof(float *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, ldb_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, beta_array, sizeof(float *)*group_count, true);
    rpc_write(client, Carray, sizeof(float *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, ldc_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, &group_count, sizeof(group_count));
    rpc_write(client, group_size, sizeof(int64_t *)*group_count, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cublasStatus_t cublasDgemmGroupedBatched(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int m_array[], const int n_array[], const int k_array[], const double alpha_array[], const double* const Aarray[], const int lda_array[], const double* const Barray[], const int ldb_array[], const double beta_array[], double* const Carray[], const int ldc_array[], int group_count, const int group_size[]) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemmGroupedBatched called" << std::endl;
#endif
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgemmGroupedBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, transa_array, sizeof(cublasOperation_t *)*group_count, true);
    rpc_write(client, transb_array, sizeof(cublasOperation_t *)*group_count, true);
    rpc_write(client, m_array, sizeof(int *)*group_count, true);
    rpc_write(client, n_array, sizeof(int *)*group_count, true);
    rpc_write(client, k_array, sizeof(int *)*group_count, true);
    rpc_write(client, alpha_array, sizeof(double *)*group_count, true);
    rpc_write(client, Aarray, sizeof(double *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, lda_array, sizeof(int *)*group_count, true);
    rpc_write(client, Barray, sizeof(double *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, ldb_array, sizeof(int *)*group_count, true);
    rpc_write(client, beta_array, sizeof(double *)*group_count, true);
    rpc_write(client, Carray, sizeof(double *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, ldc_array, sizeof(int *)*group_count, true);
    rpc_write(client, &group_count, sizeof(group_count));
    rpc_write(client, group_size, sizeof(int *)*group_count, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cublasStatus_t cublasDgemmGroupedBatched_64(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int64_t m_array[], const int64_t n_array[], const int64_t k_array[], const double alpha_array[], const double* const Aarray[], const int64_t lda_array[], const double* const Barray[], const int64_t ldb_array[], const double beta_array[], double* const Carray[], const int64_t ldc_array[], int64_t group_count, const int64_t group_size[]) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgemmGroupedBatched_64 called" << std::endl;
#endif
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgemmGroupedBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, transa_array, sizeof(cublasOperation_t *)*group_count, true);
    rpc_write(client, transb_array, sizeof(cublasOperation_t *)*group_count, true);
    rpc_write(client, m_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, n_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, k_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, alpha_array, sizeof(double *)*group_count, true);
    rpc_write(client, Aarray, sizeof(double *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, lda_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, Barray, sizeof(double *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, ldb_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, beta_array, sizeof(double *)*group_count, true);
    rpc_write(client, Carray, sizeof(double *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, ldc_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, &group_count, sizeof(group_count));
    rpc_write(client, group_size, sizeof(int64_t *)*group_count, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cublasStatus_t cublasGemmGroupedBatchedEx(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int m_array[], const int n_array[], const int k_array[], const void *alpha_array, const void* const Aarray[], cudaDataType_t Atype, const int lda_array[], const void* const Barray[], cudaDataType_t Btype, const int ldb_array[], const void *beta_array, void* const Carray[], cudaDataType_t Ctype, const int ldc_array[], int group_count, const int group_size[], cublasComputeType_t computeType) {
#ifdef DEBUG
    std::cout << "Hook: cublasGemmGroupedBatchedEx called" << std::endl;
#endif
    void *_0alpha_array = mem2server((void *)alpha_array, 0);
    void *_0beta_array = mem2server((void *)beta_array, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGemmGroupedBatchedEx);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, transa_array, sizeof(cublasOperation_t *)*group_count, true);
    rpc_write(client, transb_array, sizeof(cublasOperation_t *)*group_count, true);
    rpc_write(client, m_array, sizeof(int *)*group_count, true);
    rpc_write(client, n_array, sizeof(int *)*group_count, true);
    rpc_write(client, k_array, sizeof(int *)*group_count, true);
    rpc_write(client, &_0alpha_array, sizeof(_0alpha_array));
    rpc_write(client, Aarray, sizeof(void *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, lda_array, sizeof(int *)*group_count, true);
    rpc_write(client, Barray, sizeof(void *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, &Btype, sizeof(Btype));
    rpc_write(client, ldb_array, sizeof(int *)*group_count, true);
    rpc_write(client, &_0beta_array, sizeof(_0beta_array));
    rpc_write(client, Carray, sizeof(void *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, ldc_array, sizeof(int *)*group_count, true);
    rpc_write(client, &group_count, sizeof(group_count));
    rpc_write(client, group_size, sizeof(int *)*group_count, true);
    rpc_write(client, &computeType, sizeof(computeType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha_array, 0);
    mem2client((void *)beta_array, 0);
    return _result;
}

extern "C" cublasStatus_t cublasGemmGroupedBatchedEx_64(cublasHandle_t handle, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[], const int64_t m_array[], const int64_t n_array[], const int64_t k_array[], const void *alpha_array, const void* const Aarray[], cudaDataType_t Atype, const int64_t lda_array[], const void* const Barray[], cudaDataType_t Btype, const int64_t ldb_array[], const void *beta_array, void* const Carray[], cudaDataType_t Ctype, const int64_t ldc_array[], int64_t group_count, const int64_t group_size[], cublasComputeType_t computeType) {
#ifdef DEBUG
    std::cout << "Hook: cublasGemmGroupedBatchedEx_64 called" << std::endl;
#endif
    void *_0alpha_array = mem2server((void *)alpha_array, 0);
    void *_0beta_array = mem2server((void *)beta_array, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasGemmGroupedBatchedEx_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, transa_array, sizeof(cublasOperation_t *)*group_count, true);
    rpc_write(client, transb_array, sizeof(cublasOperation_t *)*group_count, true);
    rpc_write(client, m_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, n_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, k_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, &_0alpha_array, sizeof(_0alpha_array));
    rpc_write(client, Aarray, sizeof(void *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, &Atype, sizeof(Atype));
    rpc_write(client, lda_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, Barray, sizeof(void *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, &Btype, sizeof(Btype));
    rpc_write(client, ldb_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, &_0beta_array, sizeof(_0beta_array));
    rpc_write(client, Carray, sizeof(void *)*sum_group((int *)group_size, group_count), true);
    rpc_write(client, &Ctype, sizeof(Ctype));
    rpc_write(client, ldc_array, sizeof(int64_t *)*group_count, true);
    rpc_write(client, &group_count, sizeof(group_count));
    rpc_write(client, group_size, sizeof(int64_t *)*group_count, true);
    rpc_write(client, &computeType, sizeof(computeType));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha_array, 0);
    mem2client((void *)beta_array, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float *alpha, const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgeam called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgeam);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const float *alpha, const float *A, int64_t lda, const float *beta, const float *B, int64_t ldb, float *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgeam_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgeam_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double *alpha, const double *A, int lda, const double *beta, const double *B, int ldb, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgeam called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgeam);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const double *alpha, const double *A, int64_t lda, const double *beta, const double *B, int64_t ldb, double *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgeam_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgeam_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, const cuComplex *B, int ldb, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgeam called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgeam);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const cuComplex *alpha, const cuComplex *A, int64_t lda, const cuComplex *beta, const cuComplex *B, int64_t ldb, cuComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgeam_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgeam_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgeam called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgeam);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *beta, const cuDoubleComplex *B, int64_t ldb, cuDoubleComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgeam_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    void *_0A = mem2server((void *)A, 0);
    void *_0beta = mem2server((void *)beta, 0);
    void *_0B = mem2server((void *)B, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgeam_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0beta, sizeof(_0beta));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    mem2client((void *)A, 0);
    mem2client((void *)beta, 0);
    mem2client((void *)B, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float* const A[], int lda, float* const B[], int ldb, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrsmBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStrsmBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, A, sizeof(float *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, B, sizeof(float *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float *alpha, const float* const A[], int64_t lda, float* const B[], int64_t ldb, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrsmBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStrsmBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, A, sizeof(float *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, B, sizeof(float *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double* const A[], int lda, double* const B[], int ldb, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrsmBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtrsmBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, A, sizeof(double *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, B, sizeof(double *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double *alpha, const double* const A[], int64_t lda, double* const B[], int64_t ldb, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrsmBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtrsmBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, A, sizeof(double *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, B, sizeof(double *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex* const A[], int lda, cuComplex* const B[], int ldb, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrsmBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtrsmBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, A, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, B, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex *alpha, const cuComplex* const A[], int64_t lda, cuComplex* const B[], int64_t ldb, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrsmBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtrsmBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, A, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, B, sizeof(cuComplex *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const B[], int ldb, int batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrsmBatched called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtrsmBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, A, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, B, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex *alpha, const cuDoubleComplex* const A[], int64_t lda, cuDoubleComplex* const B[], int64_t ldb, int64_t batchCount) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrsmBatched_64 called" << std::endl;
#endif
    void *_0alpha = mem2server((void *)alpha, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtrsmBatched_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &side, sizeof(side));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &diag, sizeof(diag));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0alpha, sizeof(_0alpha));
    rpc_write(client, A, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, B, sizeof(cuDoubleComplex *)*batchCount, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &batchCount, sizeof(batchCount));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)alpha, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const float *A, int lda, const float *x, int incx, float *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSdgmm called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSdgmm);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const float *A, int64_t lda, const float *x, int64_t incx, float *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasSdgmm_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSdgmm_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const double *A, int lda, const double *x, int incx, double *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDdgmm called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDdgmm);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const double *A, int64_t lda, const double *x, int64_t incx, double *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasDdgmm_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDdgmm_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCdgmm called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCdgmm);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const cuComplex *A, int64_t lda, const cuComplex *x, int64_t incx, cuComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasCdgmm_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCdgmm_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex *C, int ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZdgmm called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZdgmm);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const cuDoubleComplex *A, int64_t lda, const cuDoubleComplex *x, int64_t incx, cuDoubleComplex *C, int64_t ldc) {
#ifdef DEBUG
    std::cout << "Hook: cublasZdgmm_64 called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0x = mem2server((void *)x, 0);
    void *_0C = mem2server((void *)C, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZdgmm_64);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &mode, sizeof(mode));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0x, sizeof(_0x));
    rpc_write(client, &incx, sizeof(incx));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)x, 0);
    mem2client((void *)C, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSmatinvBatched(cublasHandle_t handle, int n, const float* const A[], int lda, float* const Ainv[], int lda_inv, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasSmatinvBatched called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSmatinvBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, A, sizeof(float *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Ainv, sizeof(float *)*batchSize, true);
    rpc_write(client, &lda_inv, sizeof(lda_inv));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasDmatinvBatched(cublasHandle_t handle, int n, const double* const A[], int lda, double* const Ainv[], int lda_inv, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasDmatinvBatched called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDmatinvBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, A, sizeof(double *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Ainv, sizeof(double *)*batchSize, true);
    rpc_write(client, &lda_inv, sizeof(lda_inv));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasCmatinvBatched(cublasHandle_t handle, int n, const cuComplex* const A[], int lda, cuComplex* const Ainv[], int lda_inv, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasCmatinvBatched called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCmatinvBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, A, sizeof(cuComplex *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Ainv, sizeof(cuComplex *)*batchSize, true);
    rpc_write(client, &lda_inv, sizeof(lda_inv));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasZmatinvBatched(cublasHandle_t handle, int n, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const Ainv[], int lda_inv, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasZmatinvBatched called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZmatinvBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, A, sizeof(cuDoubleComplex *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Ainv, sizeof(cuDoubleComplex *)*batchSize, true);
    rpc_write(client, &lda_inv, sizeof(lda_inv));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgeqrfBatched called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgeqrfBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, Aarray, sizeof(float *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, TauArray, sizeof(float *)*batchSize, true);
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle, int m, int n, double* const Aarray[], int lda, double* const TauArray[], int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgeqrfBatched called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgeqrfBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, Aarray, sizeof(double *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, TauArray, sizeof(double *)*batchSize, true);
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, cuComplex* const Aarray[], int lda, cuComplex* const TauArray[], int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgeqrfBatched called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgeqrfBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, Aarray, sizeof(cuComplex *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, TauArray, sizeof(cuComplex *)*batchSize, true);
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const TauArray[], int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgeqrfBatched called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgeqrfBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, Aarray, sizeof(cuDoubleComplex *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, TauArray, sizeof(cuDoubleComplex *)*batchSize, true);
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
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

extern "C" cublasStatus_t cublasSgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float* const Aarray[], int lda, float* const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgelsBatched called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    void *_0devInfoArray = mem2server((void *)devInfoArray, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgelsBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &nrhs, sizeof(nrhs));
    rpc_write(client, Aarray, sizeof(float *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Carray, sizeof(float *)*batchSize, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &_0devInfoArray, sizeof(_0devInfoArray));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)info, 0);
    mem2client((void *)devInfoArray, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double* const Aarray[], int lda, double* const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgelsBatched called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    void *_0devInfoArray = mem2server((void *)devInfoArray, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgelsBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &nrhs, sizeof(nrhs));
    rpc_write(client, Aarray, sizeof(double *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Carray, sizeof(double *)*batchSize, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &_0devInfoArray, sizeof(_0devInfoArray));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)info, 0);
    mem2client((void *)devInfoArray, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuComplex* const Aarray[], int lda, cuComplex* const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgelsBatched called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    void *_0devInfoArray = mem2server((void *)devInfoArray, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgelsBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &nrhs, sizeof(nrhs));
    rpc_write(client, Aarray, sizeof(cuComplex *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Carray, sizeof(cuComplex *)*batchSize, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &_0devInfoArray, sizeof(_0devInfoArray));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)info, 0);
    mem2client((void *)devInfoArray, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgelsBatched called" << std::endl;
#endif
    void *_0info = mem2server((void *)info, 0);
    void *_0devInfoArray = mem2server((void *)devInfoArray, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgelsBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &nrhs, sizeof(nrhs));
    rpc_write(client, Aarray, sizeof(cuDoubleComplex *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, Carray, sizeof(cuDoubleComplex *)*batchSize, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &_0devInfoArray, sizeof(_0devInfoArray));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)info, 0);
    mem2client((void *)devInfoArray, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *AP, float *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasStpttr called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStpttr);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *AP, double *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtpttr called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtpttr);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *AP, cuComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtpttr called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtpttr);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *AP, cuDoubleComplex *A, int lda) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtpttr called" << std::endl;
#endif
    void *_0AP = mem2server((void *)AP, 0);
    void *_0A = mem2server((void *)A, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtpttr);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)AP, 0);
    mem2client((void *)A, 0);
    return _result;
}

extern "C" cublasStatus_t cublasStrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *A, int lda, float *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasStrttp called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasStrttp);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *A, int lda, double *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasDtrttp called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDtrttp);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *A, int lda, cuComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasCtrttp called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCtrttp);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *AP) {
#ifdef DEBUG
    std::cout << "Hook: cublasZtrttp called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, 0);
    void *_0AP = mem2server((void *)AP, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZtrttp);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &uplo, sizeof(uplo));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0AP, sizeof(_0AP));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, 0);
    mem2client((void *)AP, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n, float* const A[], int lda, int *P, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgetrfBatched called" << std::endl;
#endif
    void *_0P = mem2server((void *)P, 0);
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgetrfBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, A, sizeof(float *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0P, sizeof(_0P));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)P, 0);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n, double* const A[], int lda, int *P, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgetrfBatched called" << std::endl;
#endif
    void *_0P = mem2server((void *)P, 0);
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgetrfBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, A, sizeof(double *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0P, sizeof(_0P));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)P, 0);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex* const A[], int lda, int *P, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgetrfBatched called" << std::endl;
#endif
    void *_0P = mem2server((void *)P, 0);
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgetrfBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, A, sizeof(cuComplex *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0P, sizeof(_0P));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)P, 0);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex* const A[], int lda, int *P, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgetrfBatched called" << std::endl;
#endif
    void *_0P = mem2server((void *)P, 0);
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgetrfBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, A, sizeof(cuDoubleComplex *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0P, sizeof(_0P));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)P, 0);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgetriBatched(cublasHandle_t handle, int n, const float* const A[], int lda, const int *P, float* const C[], int ldc, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgetriBatched called" << std::endl;
#endif
    void *_0P = mem2server((void *)P, 0);
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgetriBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, A, sizeof(float *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0P, sizeof(_0P));
    rpc_write(client, C, sizeof(float *)*batchSize, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)P, 0);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgetriBatched(cublasHandle_t handle, int n, const double* const A[], int lda, const int *P, double* const C[], int ldc, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgetriBatched called" << std::endl;
#endif
    void *_0P = mem2server((void *)P, 0);
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgetriBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, A, sizeof(double *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0P, sizeof(_0P));
    rpc_write(client, C, sizeof(double *)*batchSize, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)P, 0);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgetriBatched(cublasHandle_t handle, int n, const cuComplex* const A[], int lda, const int *P, cuComplex* const C[], int ldc, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgetriBatched called" << std::endl;
#endif
    void *_0P = mem2server((void *)P, 0);
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgetriBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, A, sizeof(cuComplex *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0P, sizeof(_0P));
    rpc_write(client, C, sizeof(cuComplex *)*batchSize, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)P, 0);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgetriBatched(cublasHandle_t handle, int n, const cuDoubleComplex* const A[], int lda, const int *P, cuDoubleComplex* const C[], int ldc, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgetriBatched called" << std::endl;
#endif
    void *_0P = mem2server((void *)P, 0);
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgetriBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, A, sizeof(cuDoubleComplex *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0P, sizeof(_0P));
    rpc_write(client, C, sizeof(cuDoubleComplex *)*batchSize, true);
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)P, 0);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* const Aarray[], int lda, const int *devIpiv, float* const Barray[], int ldb, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasSgetrsBatched called" << std::endl;
#endif
    void *_0devIpiv = mem2server((void *)devIpiv, 0);
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasSgetrsBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &nrhs, sizeof(nrhs));
    rpc_write(client, Aarray, sizeof(float *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0devIpiv, sizeof(_0devIpiv));
    rpc_write(client, Barray, sizeof(float *)*batchSize, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)devIpiv, 0);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* const Aarray[], int lda, const int *devIpiv, double* const Barray[], int ldb, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasDgetrsBatched called" << std::endl;
#endif
    void *_0devIpiv = mem2server((void *)devIpiv, 0);
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasDgetrsBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &nrhs, sizeof(nrhs));
    rpc_write(client, Aarray, sizeof(double *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0devIpiv, sizeof(_0devIpiv));
    rpc_write(client, Barray, sizeof(double *)*batchSize, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)devIpiv, 0);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex* const Aarray[], int lda, const int *devIpiv, cuComplex* const Barray[], int ldb, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasCgetrsBatched called" << std::endl;
#endif
    void *_0devIpiv = mem2server((void *)devIpiv, 0);
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasCgetrsBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &nrhs, sizeof(nrhs));
    rpc_write(client, Aarray, sizeof(cuComplex *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0devIpiv, sizeof(_0devIpiv));
    rpc_write(client, Barray, sizeof(cuComplex *)*batchSize, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)devIpiv, 0);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex* const Aarray[], int lda, const int *devIpiv, cuDoubleComplex* const Barray[], int ldb, int *info, int batchSize) {
#ifdef DEBUG
    std::cout << "Hook: cublasZgetrsBatched called" << std::endl;
#endif
    void *_0devIpiv = mem2server((void *)devIpiv, 0);
    void *_0info = mem2server((void *)info, 0);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasZgetrsBatched);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &trans, sizeof(trans));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &nrhs, sizeof(nrhs));
    rpc_write(client, Aarray, sizeof(cuDoubleComplex *)*batchSize, true);
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0devIpiv, sizeof(_0devIpiv));
    rpc_write(client, Barray, sizeof(cuDoubleComplex *)*batchSize, true);
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0info, sizeof(_0info));
    rpc_write(client, &batchSize, sizeof(batchSize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)devIpiv, 0);
    mem2client((void *)info, 0);
    return _result;
}

extern "C" cublasStatus_t cublasUint8gemmBias(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, cublasOperation_t transc, int m, int n, int k, const unsigned char *A, int A_bias, int lda, const unsigned char *B, int B_bias, int ldb, unsigned char *C, int C_bias, int ldc, int C_mult, int C_shift) {
#ifdef DEBUG
    std::cout << "Hook: cublasUint8gemmBias called" << std::endl;
#endif
    void *_0A = mem2server((void *)A, sizeof(unsigned char) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    void *_0B = mem2server((void *)B, sizeof(unsigned char) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    void *_0C = mem2server((void *)C, sizeof(unsigned char) * ldc * n);
    cublasStatus_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cublasUint8gemmBias);
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &transa, sizeof(transa));
    rpc_write(client, &transb, sizeof(transb));
    rpc_write(client, &transc, sizeof(transc));
    rpc_write(client, &m, sizeof(m));
    rpc_write(client, &n, sizeof(n));
    rpc_write(client, &k, sizeof(k));
    rpc_write(client, &_0A, sizeof(_0A));
    rpc_write(client, &A_bias, sizeof(A_bias));
    rpc_write(client, &lda, sizeof(lda));
    rpc_write(client, &_0B, sizeof(_0B));
    rpc_write(client, &B_bias, sizeof(B_bias));
    rpc_write(client, &ldb, sizeof(ldb));
    rpc_write(client, &_0C, sizeof(_0C));
    rpc_write(client, &C_bias, sizeof(C_bias));
    rpc_write(client, &ldc, sizeof(ldc));
    rpc_write(client, &C_mult, sizeof(C_mult));
    rpc_write(client, &C_shift, sizeof(C_shift));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    mem2client((void *)A, sizeof(unsigned char) * lda * ((transa == CUBLAS_OP_N) ? k : n));
    mem2client((void *)B, sizeof(unsigned char) * ldb * ((transb == CUBLAS_OP_N) ? k : n));
    mem2client((void *)C, sizeof(unsigned char) * ldc * n);
    return _result;
}

