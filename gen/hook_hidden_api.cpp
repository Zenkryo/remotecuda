#include <iostream>
#include <map>
#include "../hidden_api.h"

#include "hook_api.h"
#include "client.h"
extern void *(*real_dlsym)(void *, const char *);

extern "C" void mem2server(RpcConn *conn, void **serverPtr, void *clientPtr, ssize_t size);
extern "C" void mem2client(RpcConn *conn, void *clientPtr, ssize_t size, bool del_tmp_ptr);
extern "C" void updateTmpPtr(void *clientPtr, void *serverPtr);
void *get_so_handle(const std::string &so_file);
