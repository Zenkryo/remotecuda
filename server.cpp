#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>   // for iovec
#include <arpa/inet.h> // for htonl, ntohl
#include <unistd.h>    // for read, write, close
#include <unordered_map>
#include <stdio.h>
#include <errno.h>
#include <pthread.h> // for mutex
#include <signal.h>  // for signal
#include "gen/hook_api.h"
#include "rpc.h"
#include "server.h"

#define THREAD_POOL_SIZE 10

// RPC服务器端结构
typedef struct _RpcServer {
    int listenfd;                        // 监听的文件描述符
    pthread_t threads[THREAD_POOL_SIZE]; // 线程池，用于处理多个客户端连接
    int thread_count;                    // 当前线程数
    int thread_in_use[THREAD_POOL_SIZE]; // 标记线程是否正在使用, 其值为客户端连接的文件描述符
    pthread_mutex_t mutex;               // 互斥锁，保护线程池
    uint16_t version_key;
} RpcServer;

#pragma pack(push, 1) // 按 1 字节对齐

typedef struct _ReqHeader {
    uint32_t funcId;
} ReqHeader;

#pragma pack(pop) // 恢复默认对齐
extern std::unordered_map<uint32_t, RequestHandler> handlerMap;

RequestHandler get_handler(const uint32_t funcId) {
    auto it = handlerMap.find(funcId);
    if(it != handlerMap.end()) {
        return it->second;
    }
    return nullptr;
}

// 绑定并监听端口
int rpc_bind(RpcServer *server, uint16_t port) {
    server->listenfd = socket(AF_INET, SOCK_STREAM, 0);
    if(server->listenfd < 0) {
        perror("Failed to create socket");
        return -1;
    }

    // 设置 SO_REUSEADDR 选项
    int reuse = 1;
    if(setsockopt(server->listenfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        perror("Failed to set SO_REUSEADDR");
        close(server->listenfd);
        return -1;
    }

    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    if(bind(server->listenfd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
        perror("Failed to bind socket");
        close(server->listenfd);
        return -1;
    }

    if(listen(server->listenfd, 5) < 0) {
        perror("Failed to listen on socket");
        close(server->listenfd);
        return -1;
    }

    pthread_mutex_init(&server->mutex, NULL);
    return 0;
}

// 处理客户端请求
void *rpc_handle_client(void *arg) {
    RpcServer *server = (RpcServer *)arg;
    int connfd = -1;
    ssize_t bytes_read;
    ssize_t bytes_write;
    // 找到当前线程的索引
    int thread_index = -1;
    pthread_mutex_lock(&server->mutex);
    for(int i = 0; i < THREAD_POOL_SIZE; i++) {
        if(pthread_equal(server->threads[i], pthread_self())) {
            thread_index = i;
            break;
        }
    }
    if(thread_index == -1) {
        pthread_mutex_unlock(&server->mutex);
        fprintf(stderr, "Thread not found in pool\n");
        return NULL;
    }
    connfd = server->thread_in_use[thread_index]; // 获取分配给当前线程的连接
    pthread_mutex_unlock(&server->mutex);

    // 接收客户端连接

    RpcClient client;
    client.iov_read_count = 0;
    client.iov_read2_count = 0;
    client.iov_send_count = 0;
    client.iov_send2_count = 0;
    client.sockfd = connfd;
    handshake_request handshake_req;
    handshake_response handshake_rsp;
    bytes_read = read(client.sockfd, &handshake_req, sizeof(handshake_req));
    if(bytes_read != sizeof(handshake_req)) {
        goto DONE;
    }

    if(handshake_req.version_key != server->version_key) {
        handshake_rsp.status = 1;
    } else {
        handshake_rsp.status = 0;
    }
    bytes_write = write(client.sockfd, &handshake_rsp, sizeof(handshake_rsp));
    if(bytes_write != sizeof(handshake_rsp)) {
        goto DONE;
    }
    if(handshake_rsp.status == 1) {
        goto DONE;
    }
    printf("Client connected: %d\n", connfd);

    while(1) {
        // 读取客户端请求, 先读取函数ID
        ReqHeader header;
        bytes_read = read(connfd, &header, sizeof(header));
        if(bytes_read <= 0) {
            if(bytes_read == 0) {
                printf("Client disconnected: %d\n", connfd);
            } else {
                perror("Failed to read request");
            }
            break;
        } else if(bytes_read != sizeof(header)) {
            fprintf(stderr, "Invalid request header\n");
            break;
        }
#ifdef DUMP
        hexdump("==> ", &header, sizeof(header));
#endif
        client.funcId = header.funcId;
        // 取得函数ID对应的处理函数
        RequestHandler handler = get_handler(client.funcId);
        if(handler == NULL) {
            fprintf(stderr, "No handler found for function ID: %x\n", client.funcId);
            break;
        }
        handler(&client);
        client.iov_read_count = 0;
        client.iov_read2_count = 0;
        client.iov_send_count = 0;
        client.iov_send2_count = 0;
    }
DONE:
    close(connfd);

    // 线程完成任务，标记为未使用
    pthread_mutex_lock(&server->mutex);
    server->thread_in_use[thread_index] = 0; // 标记线程为空闲
    server->thread_count--;                  // 减少线程计数
    pthread_mutex_unlock(&server->mutex);

    return NULL;
}

// 查找一个空闲的线程
int find_free_server_thread(RpcServer *server) {
    if(server->thread_count >= THREAD_POOL_SIZE) {
        return -1;
    }
    for(int i = 0; i < THREAD_POOL_SIZE; i++) {
        if(!server->thread_in_use[i]) {
            return i;
        }
    }
    return -1;
}

// 接受客户端连接并创建线程处理
void *rpc_accept_clients(void *arg) {
    RpcServer *server = (RpcServer *)arg;
    while(1) {
        int connfd = accept(server->listenfd, NULL, NULL);
        if(connfd < 0) {
            perror("Failed to accept connection");
            continue;
        }
        printf("New client connected: %d\n", connfd);

        pthread_mutex_lock(&server->mutex);

        // 找到一个空闲的线程
        int thread_index = find_free_server_thread(server);

        if(thread_index == -1) {
            pthread_mutex_unlock(&server->mutex);
            fprintf(stderr, "No available thread in pool\n");
            close(connfd);
            continue;
        }

        // 分配连接给线程
        server->thread_in_use[thread_index] = connfd;
        server->thread_count++;

        if(pthread_create(&server->threads[thread_index], NULL, rpc_handle_client, server) != 0) {
            server->thread_in_use[thread_index] = 0;
            server->thread_count--;
            pthread_mutex_unlock(&server->mutex);
            perror("Failed to create thread");
            close(connfd);
            continue;
        }

        pthread_mutex_unlock(&server->mutex);
    }
    return NULL;
}

// 关闭服务器
void rpc_close_server(RpcServer *server) {
    if(server->listenfd >= 0) {
        close(server->listenfd);
        server->listenfd = -1;
    }
    for(int i = 0; i < THREAD_POOL_SIZE; i++) {
        if(server->thread_in_use[i]) {
            pthread_join(server->threads[i], NULL);
        }
    }
    pthread_mutex_destroy(&server->mutex);
}

int main(int argc, char *argv[]) {
    RpcServer server;
    signal(SIGPIPE, SIG_IGN);

    memset(&server, 0, sizeof(RpcServer));
    server.version_key = VERSION_KEY;
    if(rpc_bind(&server, 12345) < 0) {
        return EXIT_FAILURE;
    }
    printf("Server is running on port 12345...\n");

    // 启动服务器线程
    pthread_t server_thread;
    if(pthread_create(&server_thread, NULL, rpc_accept_clients, &server) != 0) {
        perror("Failed to start server thread");
        rpc_close_server(&server);
        return EXIT_FAILURE;
    }

    // 等待服务器线程结束
    pthread_join(server_thread, NULL);

    // 关闭服务器
    rpc_close_server(&server);
    return EXIT_SUCCESS;
}
