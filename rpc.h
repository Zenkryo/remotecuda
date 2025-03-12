#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>   // for iovec
#include <arpa/inet.h> // for htonl, ntohl
#include <unistd.h>    // for read, write, close
#include <stdio.h>
#include <errno.h>
#include <pthread.h> // for mutex

#define MAX_IOCV_COUNT 32
#define MAX_CONNECTIONS 5

// RPC客户端结构
typedef struct _RpcClient {
    int sockfd;                            // TCP连接的文件描述符
    uint32_t funcId;                       // 函数ID
    struct iovec iov_send[MAX_IOCV_COUNT]; // iovec数组，用于数据传输
    struct iovec iov_read[MAX_IOCV_COUNT]; // iovec数组，用于数据传输
    int iov_send_count;                    // iovec数组的长度
    int iov_read_count;                    // iovec数组的长度
    int in_use;                            // 标记客户端是否正在使用
} RpcClient;

void rpc_init();
RpcClient *rpc_get_client();
void rpc_free_client(RpcClient *client);
void rpc_release_client(RpcClient *client);
void rpc_prepare_request(RpcClient *client, uint32_t funcId);
void rpc_write(RpcClient *client, const void *data, size_t len);
void rpc_read(RpcClient *client, void *buffer, size_t len);
int rpc_submit_request(RpcClient *client);
int rpc_prepare_response(RpcClient *client);
int rpc_submit_response(RpcClient *client);
ssize_t rpc_read_now(RpcClient *client, void *buffer, size_t len);
ssize_t rpc_read_now2(RpcClient *client, void **buffer, int *size);
void hexdump(char *desc, void *buf, size_t len);
