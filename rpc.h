#pragma once
#include <set>
#include <queue>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>   // for iovec
#include <arpa/inet.h> // for htonl, ntohl
#include <unistd.h>    // for read, write, close
#include <stdio.h>
#include <errno.h>
#include <uuid/uuid.h>
#include <pthread.h> // for mutex

#define MAX_IOCV_COUNT 64
#define MAX_CONNECTIONS 1

// RPC客户端结构
typedef struct _RpcClient {
    int sockfd;                             // TCP连接的文件描述符
    uint32_t funcId;                        // 函数ID
    struct iovec iov_send[MAX_IOCV_COUNT];  // iovec数组，用于预先确定长度的数据传输
    struct iovec iov_send2[MAX_IOCV_COUNT]; // iovec数组，用长度不预先确定的数据传输
    struct iovec iov_read[MAX_IOCV_COUNT];  // iovec数组，用于预先确定长度的数据传输
    struct iovec iov_read2[MAX_IOCV_COUNT]; // iovec数组，用于长度不预先确定的数据传输
    int iov_send_count;                     // iov_send数组的长度
    int iov_send2_count;                    // iov_send2数组的长度
    int iov_read_count;                     // iov_read数组的长度
    int iov_read2_count;                    // iov_read2数组的长度
    int in_use;                             // 标记客户端是否正在使用
    std::queue<void *> *tmpbufs;            // 用于存储需要释放的缓冲区
} RpcClient;

typedef struct {
    uuid_t id;
    uint16_t version_key;
} handshake_request;

typedef struct {
    int status;
} handshake_response;

void rpc_init(uint16_t version_key);
RpcClient *rpc_get_client();
void rpc_free_client(RpcClient *client);
void rpc_release_client(RpcClient *client);
void rpc_prepare_request(RpcClient *client, uint32_t funcId);
void rpc_write(RpcClient *client, const void *data, size_t len, bool with_len = false);
void rpc_read(RpcClient *client, void *buffer, size_t len, bool with_len = false);
int rpc_submit_request(RpcClient *client);
int rpc_prepare_response(RpcClient *client);
int rpc_submit_response(RpcClient *client);
ssize_t read_all_now(RpcClient *client, void **buffer, int *size, int count);
ssize_t read_one_now(RpcClient *client, void *buffer, int size, bool with_len);
void hexdump(const char *desc, void *buf, size_t len);
int sum_group(int *group_size, int group_count);
