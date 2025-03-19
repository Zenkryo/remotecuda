#include <iostream>
#include <cstdlib>
#include <dlfcn.h>
#include <unordered_map>
#include <string>
#include "rpc.h"

void *getHookFunc(const char *symbol);
// 保存系统的 dlsym 函数指针
void *(*real_dlsym)(void *, const char *) = nullptr;

void *(*real_dlopen)(const char *, int) = nullptr;

extern "C" void *dlopen(const char *filename, int flag) {
#ifdef DEBUG
    std::cout << "dlopen " << filename << std::endl;
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
    std::cout << "dlsym: " << symbol << std::endl;
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

static std::unordered_map<std::string, void *> so_handles;

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
    for(auto &handle : so_handles) {
        if(handle.second) {
            dlclose(handle.second);
        }
    }
}

// 构造函数和析构函数
__attribute__((constructor)) void init() { init_hook(); }

__attribute__((destructor)) void cleanup() { cleanup_hook(); }
