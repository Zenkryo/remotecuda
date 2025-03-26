#include <stdlib.h>
#include <iostream>
#include "rpc.h"

RpcClient clients_pool[MAX_CONNECTIONS];
pthread_mutex_t pool_lock;
pthread_t rpc_thread_id;
uuid_t client_id;

// 连接服务器
static int rpc_connect(const char *server, uint16_t port, uint16_t version_key) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if(sockfd < 0) {
        perror("socket creation failed");
        return -1;
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, server, &server_addr.sin_addr);

    if(connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect failed");
        close(sockfd);
        return -1;
    }

    handshake_request handshake_req;
    uuid_copy(handshake_req.id, client_id);
    handshake_req.version_key = version_key;

    // 发送握手请求
    if(write(sockfd, &handshake_req, sizeof(handshake_req)) != sizeof(handshake_req)) {
        perror("registration request failed");
        close(sockfd);
        return -1;
    }

    // 读取握手响应
    handshake_response handshake_rsp;
    if(read(sockfd, &handshake_rsp, sizeof(handshake_rsp)) != sizeof(handshake_rsp)) {
        perror("registration response failed");
        close(sockfd);
        return -1;
    }
    if(handshake_rsp.status != 0) {
        perror("registration failed");
        close(sockfd);
        return -1;
    }
    return sockfd;
}

//  线程函数，用于维护连接
static void *rpc_thread(void *arg) {
    uint16_t version_key = *(uint16_t *)arg;
    delete(uint16_t *)arg;
    // 获取环境变量CUDA_SERVER
    const char *cuda_server = getenv("CUDA_SERVER");
    if(!cuda_server) {
        cuda_server = "127.0.0.1"; // fallback
    }

    while(1) {
        pthread_mutex_lock(&pool_lock);
        for(int i = 0; i < MAX_CONNECTIONS; i++) {
            if(clients_pool[i].sockfd == -1) {
                int sockfd = rpc_connect(cuda_server, 12345, version_key);
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
        shutdown(client->sockfd, SHUT_RDWR);
        close(client->sockfd);
        client->sockfd = -1;
    }
    pthread_mutex_unlock(&pool_lock); // 解锁
}

static ssize_t write_full_iovec(int sockfd, struct iovec *iov, int iovcnt) {
    if(iovcnt == 0) {
        return 0;
    }
    ssize_t total_bytes = 0; // 总共写入的字节数
    while(iovcnt > 0) {
        // 使用 writev 将数据写入 socket
        ssize_t bytes_written = writev(sockfd, iov, iovcnt);

        if(bytes_written < 0) {
            // 写入错误
            if(errno == EINTR) {
                // 被信号中断，继续尝试写入
                continue;
            }
            return -1; // 其他错误返回-1
        }

        if(bytes_written == 0) {
            // 连接关闭（通常不会发生，但为健壮性考虑）
            return -1;
        }

        // 更新总字节数
        total_bytes += bytes_written;
#ifdef DUMP
        // 调试输出（可选，仿照 read_full_iovec）
        ssize_t bytes_remaining = bytes_written;
        for(int i = 0; i < iovcnt && bytes_remaining > 0; i++) {
            int len = iov[i].iov_len > bytes_remaining ? bytes_remaining : iov[i].iov_len;
            hexdump("<== ", iov[i].iov_base, len);
            bytes_remaining -= len;
        }
#endif
        // 调整 iovec 数组以处理部分写入的情况
        while(bytes_written > 0 && iovcnt > 0) {
            if(bytes_written >= iov->iov_len) {
                // 当前缓冲区已全部写入，移动到下一个
                bytes_written -= iov->iov_len;
                iov++;
                iovcnt--;
            } else {
                // 当前缓冲区未写完，更新指针和长度
                iov->iov_base = (char *)iov->iov_base + bytes_written;
                iov->iov_len -= bytes_written;
                bytes_written = 0;
            }
        }
    }
    return total_bytes;
}

static ssize_t read_full_iovec(int sockfd, struct iovec *iov, int iovcnt) {
    ssize_t total_bytes = 0; // 总共读取的字节数

    while(iovcnt > 0) {
        // 使用 readv 从 socket 读取数据
        ssize_t bytes_read = readv(sockfd, iov, iovcnt);
        if(bytes_read < 0) {
            // 读取错误
            if(errno == EINTR) {
                // 被信号中断，继续尝试读取
                continue;
            }
            return -1; // 其他错误返回-1
        }

        if(bytes_read == 0) {
            // 连接关闭
            return -1;
        }

        // 更新总字节数
        total_bytes += bytes_read;

#ifdef DUMP
        ssize_t bytes_remaining = bytes_read;

        for(int i = 0; i < iovcnt && bytes_remaining > 0; i++) {
            int len = iov[i].iov_len > bytes_remaining ? bytes_remaining : iov[i].iov_len;
            hexdump("==> ", iov[i].iov_base, len);
            bytes_remaining -= len;
        }
#endif
        // 调整 iovec 数组以处理部分读取的情况
        while(bytes_read > 0 && iovcnt > 0) {
            if(bytes_read >= iov->iov_len) {
                // 当前缓冲区已满，移动到下一个
                bytes_read -= iov->iov_len;
                iov++;
                iovcnt--;
            } else {
                // 当前缓冲区未满，更新指针和长度
                iov->iov_base = (char *)iov->iov_base + bytes_read;
                iov->iov_len -= bytes_read;
                bytes_read = 0;
            }
        }
    }
    return total_bytes;
}

// 发送client里待发送数据，支持发送数据和长度+数据两种模式
static ssize_t writev_all(RpcClient *client, bool with_len = false) {
    uint32_t lengths[MAX_IOCV_COUNT];
    struct iovec iov_with_len[MAX_IOCV_COUNT * 2];

    struct iovec *iov2send = client->iov_send;
    int iov2send_count = client->iov_send_count;

    if(with_len) {
        iov2send_count = client->iov_send2_count * 2;
        iov2send = iov_with_len;
        for(int i = 0; i < client->iov_send2_count; i++) {
            lengths[i] = htonl((uint32_t)client->iov_send2[i].iov_len);
            iov_with_len[2 * i].iov_base = &lengths[i];
            iov_with_len[2 * i].iov_len = sizeof(uint32_t);
            iov_with_len[2 * i + 1].iov_base = client->iov_send2[i].iov_base;
            iov_with_len[2 * i + 1].iov_len = client->iov_send2[i].iov_len;
        }
    }
    return write_full_iovec(client->sockfd, iov2send, iov2send_count);
}

// 读取数据到client的缓冲区，支持读取数据和长度+数据两种模式
static ssize_t readv_all(RpcClient *client, bool with_len = false) {
    ssize_t total_read = 0; // 已读取的总字节数
    ssize_t bytes_read;     // 每次读取的字节数

    std::set<void *> buffers;
    if(with_len) {
        for(int i = 0; i < client->iov_read2_count; i++) {
            // 先读取长度字段
            struct iovec iov_for_len;
            uint32_t length;
            iov_for_len.iov_base = &length;
            iov_for_len.iov_len = sizeof(length);
            bytes_read = read_full_iovec(client->sockfd, &iov_for_len, 1);
            if(bytes_read < 0) {
                goto ERR;
            }
            total_read += bytes_read;
            length = ntohl(length);
            if(client->iov_read2[i].iov_len > 0 && length > client->iov_read2[i].iov_len) {
                errno = ENOBUFS; // 缓冲区不足
                goto ERR;
            }
            // iov_len为0表示需要动态分配缓冲区
            if(client->iov_read2[i].iov_len == 0) {
                void **buffer = (void **)client->iov_read2[i].iov_base;
                *buffer = malloc(length);
                if(*buffer == nullptr) {
                    errno = ENOBUFS; // 缓冲区不足
                    goto ERR;
                }
                buffers.insert(*buffer);
                client->iov_read2[i].iov_base = *buffer; // 更新iov[i].iov_base
            } else if(client->iov_read2[i].iov_len < length) {
                errno = ENOBUFS; // 缓冲区不足
                goto ERR;
            }
            client->iov_read2[i].iov_len = length;
            bytes_read = read_full_iovec(client->sockfd, client->iov_read2 + i, 1);
            if(bytes_read < 0) {
                goto ERR;
            }
            total_read += bytes_read;
        }
    } else {
        bytes_read = read_full_iovec(client->sockfd, client->iov_read, client->iov_read_count);
        if(bytes_read < 0) {
            return -1;
        }
        total_read += bytes_read;
    }
    return total_read;
ERR:
    for(auto it = buffers.begin(); it != buffers.end(); it++) {
        free(*it);
    }
    return -1;
}

// 打印缓冲区内容
void hexdump(const char *desc, void *buf, size_t len) {
    unsigned char *p = (unsigned char *)buf;
    printf("%s len: %lu\n", desc, len);
    // 每行显示16个字节
    for(size_t i = 0; i < len; i += 16) {
        printf("%08lx: ", i);
        for(size_t j = 0; j < 16; j++) {
            if(i + j < len) {
                printf("%02x ", p[i + j]);
            } else {
                printf("   ");
            }
        }
        printf(" ");
        for(size_t j = 0; j < 16; j++) {
            if(i + j < len) {
                printf("%c", isprint(p[i + j]) ? p[i + j] : '.');
            }
        }
        printf("\n");
    }
}

// 初始化连接池
void rpc_init(uint16_t version_key) {
    uuid_generate(client_id);
    pthread_mutex_init(&pool_lock, NULL);
    for(int i = 0; i < MAX_CONNECTIONS; i++) {
        memset(&clients_pool[i], 0, sizeof(RpcClient));
        clients_pool[i].sockfd = -1;
    }
    // 创建一个线程用于维护连接
    uint16_t *key = new uint16_t(version_key);
    pthread_create(&rpc_thread_id, NULL, rpc_thread, key);
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
                clients_pool[i].iov_send2_count = 0;
                clients_pool[i].iov_read_count = 0;
                clients_pool[i].iov_read2_count = 0;
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
    client->iov_send2_count = 0;
    client->iov_read_count = 0;
    client->iov_read2_count = 0;
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

// 准备要发送的数据
void rpc_write(RpcClient *client, const void *data, size_t len, bool with_len) {
    if(with_len) {
        client->iov_send2[client->iov_send2_count].iov_base = (void *)data;
        client->iov_send2[client->iov_send2_count].iov_len = len;
        client->iov_send2_count++;
    } else {
        client->iov_send[client->iov_send_count].iov_base = (void *)data;
        client->iov_send[client->iov_send_count].iov_len = len;
        client->iov_send_count++;
    }
}

// 准备接收数据的缓冲区
void rpc_read(RpcClient *client, void *buffer, size_t len, bool with_len) {
    if(with_len) {
        client->iov_read2[client->iov_read2_count].iov_base = buffer;
        client->iov_read2[client->iov_read2_count].iov_len = len;
        client->iov_read2_count++;
    } else {
        client->iov_read[client->iov_read_count].iov_base = buffer;
        client->iov_read[client->iov_read_count].iov_len = len;
        client->iov_read_count++;
    }
}

// 读取count个带长度的数据到动态分配的缓冲区
ssize_t read_all_now(RpcClient *client, void **buffer, int *size, int count) {
    ssize_t total_read = 0; // 已读取的总字节数
    ssize_t bytes_read;     // 每次读取的字节数

    for(int i = 0; i < count; i++) {
        struct iovec iov;
        uint32_t length;
        iov.iov_base = &length;
        iov.iov_len = sizeof(length);
        bytes_read = read_full_iovec(client->sockfd, &iov, 1);
        if(bytes_read < 0) {
            return -1;
        }
        total_read += bytes_read;
        length = ntohl(length);
        iov.iov_len = length;
        iov.iov_base = malloc(length);
        if(iov.iov_base == NULL) {
            errno = ENOBUFS; // 缓冲区不足
            return -1;
        }
        bytes_read = read_full_iovec(client->sockfd, &iov, 1);
        if(bytes_read < 0) {
            free(iov.iov_base);
            return -1;
        }
        buffer[i] = iov.iov_base;
        if(size != NULL) {
            size[i] = length;
        }
        total_read += bytes_read;
    }
    return total_read;
}

// 读取1数据到以分配的缓冲区
ssize_t read_one_now(RpcClient *client, void *buffer, int size, bool with_len = false) {
    ssize_t total_read = 0; // 已读取的总字节数
    ssize_t bytes_read;     // 每次读取的字节数

    struct iovec iov;
    uint32_t length = size;
    void *tmp_buffer = nullptr;
    if(with_len) {
        iov.iov_base = &length;
        iov.iov_len = sizeof(length);
        bytes_read = read_full_iovec(client->sockfd, &iov, 1);
        if(bytes_read < 0) {
            return -1;
        }
        total_read += bytes_read;
        length = ntohl(length);
        if(size > 0 && length > size) {
            errno = ENOBUFS; // 缓冲区不足
            return -1;
        }
    }
    if(length == 0) {
        return total_read;
    }
    if(size == 0) {
        tmp_buffer = (void *)malloc(length);
        if(tmp_buffer == nullptr) {
            errno = ENOBUFS; // 缓冲区不足
            return -1;
        }
        *(void **)buffer = tmp_buffer;
    } else {
        tmp_buffer = buffer;
    }
    iov.iov_len = length;
    iov.iov_base = tmp_buffer;
    bytes_read = read_full_iovec(client->sockfd, &iov, 1);
    if(bytes_read < 0) {
        if(size == 0) {
            free(tmp_buffer);
        }
        return -1;
    }
    total_read += bytes_read;
    return total_read;
}

// 请求并等待响应
int rpc_submit_request(RpcClient *client) {
    // 发送数据
    if(writev_all(client, false) < 0 || writev_all(client, true) < 0) {
        perror("Failed to send request");
        rpc_release_client(client);
        return -1;
    }
    // 读取数据
    if(readv_all(client) < 0 || readv_all(client, true) < 0) {
        perror("Failed to read response");
        rpc_release_client(client);
        return -1;
    }
    return 0;
}

// 读取所有的RPC请求参数
int rpc_prepare_response(RpcClient *client) {
    if(readv_all(client) < 0 || readv_all(client, true) < 0) {
        perror("Failed to read request");
        return -1;
    }
    return 0;
}

// 提交RPC响应
int rpc_submit_response(RpcClient *client) {
    if(writev_all(client) < 0 || writev_all(client, true) < 0) {
        perror("Failed to send response");
        return -1;
    }
    return 0;
}

int sum_group(int *group_size, int group_count) {
    int sum = 0;
    for(int i = 0; i < group_count; i++) {
        sum += group_size[i];
    }
    return sum;
}
