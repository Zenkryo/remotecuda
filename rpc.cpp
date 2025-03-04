#include <stdlib.h>
#include <iostream>
#include "rpc.h"

RpcClient clients_pool[MAX_CONNECTIONS];
pthread_mutex_t pool_lock;
pthread_t rpc_thread_id;

// 连接服务器
static int rpc_connect(const char *server, uint16_t port) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if(sockfd < 0) {
        perror("socket");
        return -1;
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, server, &server_addr.sin_addr);

    if(connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect");
        close(sockfd);
        return -1;
    }
    return sockfd;
}

//  rpc_thread
static void *rpc_thread(void *arg) {
    while(1) {
        pthread_mutex_lock(&pool_lock);
        for(int i = 0; i < MAX_CONNECTIONS; i++) {
            if(clients_pool[i].sockfd == -1) {
                int sockfd = rpc_connect("127.0.0.1", 12345);
                if(sockfd != -1) {
                    clients_pool[i].sockfd = sockfd;
                }
            }
        }
        pthread_mutex_unlock(&pool_lock);
        sleep(1); // 每隔1秒检查一次连接
    }
    return NULL;
}

// 关闭RPC连接
static void rpc_close(RpcClient *client) {
    pthread_mutex_lock(&pool_lock); // 加锁
    if(client->sockfd >= 0) {
        close(client->sockfd);
        client->sockfd = -1;
    }
    pthread_mutex_unlock(&pool_lock); // 解锁
}

// 确保所有数据发送完毕
static ssize_t writev_all(int sockfd, struct iovec *iov, int iovcnt) {
    ssize_t total_sent = 0;
    ssize_t bytes_sent;
    struct iovec new_iov[MAX_IOCV_COUNT * 2];
    // 计算所需的总 iovec 数量（每个原始 iovec 需要两个：长度 + 数据）
    int new_iovcnt = iovcnt * 2;
    // 为每个 iovec 创建长度字段并填充新数组
    uint16_t lengths[MAX_IOCV_COUNT];
    size_t total_length = 0;
    for(int i = 0; i < iovcnt; i++) {
        printf("1===========\n");
        if(iov[i].iov_len > UINT32_MAX) {
            errno = EOVERFLOW;
            return -1;
        }
        lengths[i] = htons((uint16_t)iov[i].iov_len);
        new_iov[i * 2].iov_base = &lengths[i];
        new_iov[i * 2].iov_len = sizeof(uint16_t);
        new_iov[i * 2 + 1].iov_base = iov[i].iov_base;
        new_iov[i * 2 + 1].iov_len = iov[i].iov_len;
        total_length += sizeof(uint16_t) + iov[i].iov_len;
    }
    // 使用一次 writev 发送所有数据
    while(total_length > 0) {
        printf("2=========== %d\n", new_iovcnt);
        for(int i = 0; i < new_iovcnt; i++) {
            printf("<== ");
            for(int j = 0; j < new_iov[i].iov_len; j++) {
                printf("%02x ", ((uint8_t *)new_iov[i].iov_base)[j]);
            }
            printf("\n");
        }
        bytes_sent = writev(sockfd, new_iov, new_iovcnt);
        printf("3=========== %d\n", bytes_sent);
        if(bytes_sent < 0) {
            if(errno == EINTR) {
                continue;
            }
            return -1;
        }
        int dumped = 0;
        // dump print data just sent
        for(int i = 0; i < new_iovcnt; i++) {
            printf("<== ");
            for(int j = 0; j < new_iov[i].iov_len && dumped < bytes_sent; j++, dumped++) {
                printf("%02x ", ((uint8_t *)new_iov[i].iov_base)[j]);
            }
            printf("\n");
        }

        total_sent += bytes_sent;

        // 更新 iovec 数组以处理部分写入
        size_t bytes_remaining = bytes_sent;
        for(int i = 0; i < new_iovcnt && bytes_remaining > 0; i++) {
            if(bytes_remaining >= new_iov[i].iov_len) {
                bytes_remaining -= new_iov[i].iov_len;
                new_iov[i].iov_len = 0;
            } else {
                new_iov[i].iov_base = (char *)new_iov[i].iov_base + bytes_remaining;
                new_iov[i].iov_len -= bytes_remaining;
                bytes_remaining = 0;
            }
        }
        total_length -= bytes_sent;
    }
    if(total_length > 0) {
        return -1;
    }
    return 0;
}

// 确保所有数据读取完毕
static ssize_t readv_all(int sockfd, struct iovec *iov, int iovcnt) {
    ssize_t total_read = 0; // 已读取的总字节数
    ssize_t bytes_read;     // 每次读取的字节数
    size_t remaining = 0;
    for(int i = 0; i < iovcnt; i++) {
        // 读取长度字段 (2 字节)
        uint16_t length;
        size_t length_bytes = sizeof(length);
        uint8_t *length_ptr = (uint8_t *)&length;

        while(length_bytes > 0) {
            bytes_read = read(sockfd, length_ptr, length_bytes);
            if(bytes_read < 0) {
                if(errno == EINTR) {
                    continue;
                }
                return -1;
            }
            if(bytes_read == 0) {
                return -1; // 对端关闭连接
            }
            printf("==> ");
            for(int j = 0; j < bytes_read; j++) {
                printf("%02x ", (length_ptr)[j]);
            }
            printf("\n");
            length_ptr += bytes_read;
            length_bytes -= bytes_read;
            total_read += bytes_read;
        }

        length = ntohs(length); // 转换为主机字节序
        // 检查缓冲区是否足够大
        if(length > iov[i].iov_len) {
            errno = ENOBUFS; // 缓冲区不足
            return -1;
        }
        // 更新当前 iovec 的长度为实际数据长度
        iov[i].iov_len = length;

        // 读取数据
        remaining = length;
        uint8_t *data_ptr = (uint8_t *)iov[i].iov_base;

        while(remaining > 0) {
            bytes_read = read(sockfd, data_ptr, remaining);
            if(bytes_read < 0) {
                if(errno == EINTR) {
                    continue;
                }
                return -1;
            }
            if(bytes_read == 0) {
                return -1; // 对端关闭连接
            }

            printf("==> ");
            for(int j = 0; j < bytes_read; j++) {
                printf("%02x ", (data_ptr)[j]);
            }
            printf("\n");
            total_read += bytes_read;
            data_ptr += bytes_read;
            remaining -= bytes_read;
        }
    }
    if(remaining > 0) {
        return -1;
    }
    return 0;
}

// 初始化连接池
void rpc_init() {
    pthread_mutex_init(&pool_lock, NULL);
    for(int i = 0; i < MAX_CONNECTIONS; i++) {
        memset(&clients_pool[i], 0, sizeof(RpcClient));
        clients_pool[i].sockfd = -1;
    }
    // 创建一个线程用于维护连接
    pthread_create(&rpc_thread_id, NULL, rpc_thread, NULL);
}

// 取得一个RPC客户端
RpcClient *rpc_get_client() {
    int i = 0;
    while(i < 5) {
        pthread_mutex_lock(&pool_lock);
        for(int i = 0; i < MAX_CONNECTIONS; i++) {
            if(clients_pool[i].sockfd != -1 && !clients_pool[i].in_use) {
                clients_pool[i].in_use = 1;
                clients_pool[i].iov_send_count = 0;
                clients_pool[i].iov_read_count = 0;
                pthread_mutex_unlock(&pool_lock);
                return &clients_pool[i];
            }
        }
        pthread_mutex_unlock(&pool_lock);
        sleep(1);
    }
    return NULL;
}

// 释放RPC客户端
void rpc_free_client(RpcClient *client) {
    pthread_mutex_lock(&pool_lock);
    client->in_use = 0;
    client->iov_send_count = 0;
    client->iov_read_count = 0;
    pthread_mutex_unlock(&pool_lock);
}

// 在连接异常时释放RPC客户端
void rpc_release_client(RpcClient *client) {
    rpc_close(client);
    rpc_free_client(client);
}

// 准备一个RPC请求
void rpc_prepare_request(RpcClient *client, uint32_t funcId) {
    client->funcId = funcId;
    client->iov_send[0].iov_base = &client->funcId;
    client->iov_send[0].iov_len = sizeof(client->funcId);
    client->iov_send_count = 1;
}

// 写入数据
void rpc_write(RpcClient *client, const void *data, size_t len) {
    client->iov_send[client->iov_send_count].iov_base = (void *)data;
    client->iov_send[client->iov_send_count].iov_len = len;
    client->iov_send_count++;
}

// 注册接收数据的缓冲区
void rpc_read(RpcClient *client, void *buffer, size_t len) {
    client->iov_read[client->iov_read_count].iov_base = buffer;
    client->iov_read[client->iov_read_count].iov_len = len;
    client->iov_read_count++;
}

ssize_t rpc_read_now(RpcClient *client, void *buffer, size_t len) {
    struct iovec iov[1];
    iov[0].iov_base = buffer;
    iov[0].iov_len = len;

    return readv_all(client->sockfd, iov, 1);
}

// 请求并等待响应
int rpc_submit_request(RpcClient *client) {
    if(writev_all(client->sockfd, client->iov_send, client->iov_send_count) < 0) {
        perror("Failed to send request");
        rpc_release_client(client);
        return -1;
    }
    client->iov_send_count = 0;
    if(readv_all(client->sockfd, client->iov_read, client->iov_read_count) < 0) {
        perror("Failed to read response");
        rpc_release_client(client);
        return -1;
    }
    client->iov_read_count = 0;
    return 0;
}

// 读取所有的RPC请求参数
int rpc_prepare_response(RpcClient *client) {
    if(readv_all(client->sockfd, client->iov_read, client->iov_read_count) < 0) {
        perror("Failed to read request");
        return -1;
    }
    client->iov_read_count = 0;
    return 0;
}

// 提交RPC响应
int rpc_submit_response(RpcClient *client) {
    if(writev_all(client->sockfd, client->iov_send, client->iov_send_count) < 0) {
        perror("Failed to send response");
        return -1;
    }
    client->iov_send_count = 0;
    return 0;
}
