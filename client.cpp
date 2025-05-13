#include <iostream>
#include <cstdlib>
#include <dlfcn.h>
#include <map>
#include <string>
#include <cuda_runtime.h>
#include <cublas_api.h>
#include "rpc/rpc_core.h"
#include "gen/hook_api.h"
using namespace rpc;

void *getHookFunc(const char *symbol);

// 保存系统的 dlsym 函数指针
void *(*real_dlsym)(void *, const char *) = nullptr;

void *(*real_dlopen)(const char *, int) = nullptr;

extern "C" void *dlopen(const char *filename, int flag) {

#ifdef DEBUG
    // std::cout << "dlopen " << filename << std::endl;
#endif
    // 初始化 real_dlopen
    if(real_dlopen == nullptr) {
        real_dlopen = reinterpret_cast<void *(*)(const char *, int)>(dlvsym(RTLD_NEXT, "dlopen", "GLIBC_2.2.5"));
        if(real_dlopen == nullptr) {
            std::cerr << "Error: Failed to find real dlopen" << std::endl;
            std::exit(1);
        }
    }

    // 调用系统的 dlopen
    return real_dlopen(filename, flag);
}

extern "C" void *dlsym(void *handle, const char *symbol) {
#ifdef DEBUG
    // std::cout << "dlsym: " << symbol << std::endl;
#endif
    // 初始化 real_dlsym
    if(real_dlsym == nullptr) {
        real_dlsym = reinterpret_cast<void *(*)(void *, const char *)>(dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"));
        if(real_dlsym == nullptr) {
            std::cerr << "Error: Failed to find real dlsym" << std::endl;
            std::exit(1);
        }
    }
    void *fp = getHookFunc(symbol);
    if(fp != nullptr) {
        return fp;
    }
#ifdef DEBUG
    std::cout << "Not found in hook functions " << symbol << std::endl;
#endif
    // 如果 map 中找不到，调用系统的 dlsym
    return real_dlsym(handle, symbol);
}

static std::map<std::string, void *> so_handles;
std::unique_ptr<RpcClient> client;

void rpc_init() {
    client = std::unique_ptr<RpcClient>(new RpcClient(VERSION_KEY));
    // 连接到服务器
    RpcError err = client->connect("127.0.0.1", 12345, 5);
    if(err != RpcError::OK) {
        std::cerr << "Failed to connect: " << static_cast<int>(err) << std::endl;
        return;
    }
}
// Hook 的初始化函数
void init_hook() { rpc_init(); }

void *get_so_handle(const std::string &so_file) {
    if(so_handles.find(so_file) == so_handles.end()) {
        so_handles[so_file] = dlopen(so_file.c_str(), RTLD_LAZY);
        if(!so_handles[so_file]) {
            std::cerr << "Error: " << dlerror() << std::endl;
            std::exit(1);
        }
    }
    return so_handles[so_file];
}

void cleanup_hook() {
    // rpc_destroy();
    for(auto &handle : so_handles) {
        if(handle.second) {
            dlclose(handle.second);
        }
    }
}

// 构造函数和析构函数
__attribute__((constructor)) void init() { init_hook(); }

__attribute__((destructor)) void cleanup() { cleanup_hook(); }

int sizeofType(cudaDataType type) {
    switch(type) {
    case CUDA_R_16F:
        return 2; // 半精度浮点数
    case CUDA_C_16F:
        return 4; // 半精度复数 (2 + 2)
    case CUDA_R_16BF:
        return 2; // 半精度bfloat16
    case CUDA_C_16BF:
        return 4; // 半精度复数bfloat16 (2 + 2)
    case CUDA_R_32F:
        return 4; // 单精度浮点数
    case CUDA_C_32F:
        return 8; // 单精度复数 (4 + 4)
    case CUDA_R_64F:
        return 8; // 双精度浮点数
    case CUDA_C_64F:
        return 16; // 双精度复数 (8 + 8)
    case CUDA_R_4I:
        return 1; // 4位整数
    case CUDA_C_4I:
        return 2; // 4位复数整数 (1 + 1)
    case CUDA_R_4U:
        return 1; // 4位无符号整数
    case CUDA_C_4U:
        return 2; // 4位无符号复数整数 (1 + 1)
    case CUDA_R_8I:
        return 1; // 8位整数
    case CUDA_C_8I:
        return 2; // 8位复数整数 (1 + 1)
    case CUDA_R_8U:
        return 1; // 8位无符号整数
    case CUDA_C_8U:
        return 2; // 8位无符号复数整数 (1 + 1)
    case CUDA_R_16I:
        return 2; // 16位整数
    case CUDA_C_16I:
        return 4; // 16位复数整数 (2 + 2)
    case CUDA_R_16U:
        return 2; // 16位无符号整数
    case CUDA_C_16U:
        return 4; // 16位无符号复数整数 (2 + 2)
    case CUDA_R_32I:
        return 4; // 32位整数
    case CUDA_C_32I:
        return 8; // 32位复数整数 (4 + 4)
    case CUDA_R_32U:
        return 4; // 32位无符号整数
    case CUDA_C_32U:
        return 8; // 32位无符号复数整数 (4 + 4)
    case CUDA_R_64I:
        return 8; // 64位整数
    case CUDA_C_64I:
        return 16; // 64位复数整数 (8 + 8)
    case CUDA_R_64U:
        return 8; // 64位无符号整数
    case CUDA_C_64U:
        return 16; // 64位无符号复数整数 (8 + 8)
    default:
        printf("Unsupported cudaDataType\n");
        return -1; // 未知类型返回错误
    }
}
int sizeofPoolAttribute(int attr) {
    if(attr < 4) {
        return sizeof(int);
    }
    return sizeof(uint64_t);
}

RpcConn *rpc_get_conn() {
    RpcConn *conn = nullptr;
    int i = 10;
    while(i-- > 0) {
        conn = client->acquire_connection();

        if(conn) {
            return conn;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return nullptr;
}

void rpc_release_conn(RpcConn *conn, bool to_close) { client->release_connection(conn, to_close); }

int sum_group(int *group_size, int group_count) {
    int sum = 0;
    for(int i = 0; i < group_count; i++) {
        sum += group_size[i];
    }
    return sum;
}
