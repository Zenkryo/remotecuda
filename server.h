#ifndef SERVER_H
#define SERVER_H

#include <map>
#include <cstdint>
#include "rpc.h"
#include <uuid/uuid.h>
#include <string>
#include <functional>
#include <pthread.h>

// RPC服务器端结构
typedef struct _RpcServer {
    int listenfd;                                         // 监听的文件描述符
    uint16_t version_key;                                 // 版本号
    std::map<std::string, RpcClient *> client_async_conn; // 每个客户端有唯一一个异步通信连接
    std::vector<pthread_t> threads;                       // 所有子线程的ID
} RpcServer;

typedef int (*RequestHandler)(void *);

RequestHandler get_handler(const uint32_t funcId);

#endif // SERVER_H
