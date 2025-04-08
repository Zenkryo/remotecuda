#include <iostream>
#include <unordered_map>
#include "../hidden_api.h"

#include "hook_api.h"
#include "../rpc.h"
extern void *(*real_dlsym)(void *, const char *);

extern "C" void mem2server(RpcClient *client, void **serverPtr,void *clientPtr, ssize_t size);
extern "C" void mem2client(RpcClient *client, void *clientPtr, ssize_t size);
void *get_so_handle(const std::string &so_file);
