#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>   // for iovec
#include <arpa/inet.h> // for htonl, ntohl
#include <unistd.h>    // for read, write, close
#include <map>
#include <stdio.h>
#include <errno.h>
#include <uuid/uuid.h>
#include <pthread.h> // for mutex
#include <signal.h>  // for signal
#include "gen/hook_api.h"
#include "server.h"

#pragma pack(push, 1) // 按 1 字节对齐

typedef struct _ReqHeader {
    uint32_t funcId;
} ReqHeader;

#pragma pack(pop) // 恢复默认对齐

extern std::map<uint32_t, RequestHandler> handlerMap;

RequestHandler get_handler(const uint32_t funcId) {
    auto it = handlerMap.find(funcId);
    if(it != handlerMap.end()) {
        return it->second;
    }
    return nullptr;
}

RpcServer server;

// 绑定并监听端口
int rpc_bind(uint16_t port) {
    server.listenfd = socket(AF_INET, SOCK_STREAM, 0);
    if(server.listenfd < 0) {
        perror("Failed to create socket");
        return -1;
    }

    // 设置 SO_REUSEADDR 选项
    int reuse = 1;
    if(setsockopt(server.listenfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        perror("Failed to set SO_REUSEADDR");
        close(server.listenfd);
        return -1;
    }

    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    if(bind(server.listenfd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
        perror("Failed to bind socket");
        close(server.listenfd);
        return -1;
    }

    if(listen(server.listenfd, 5) < 0) {
        perror("Failed to listen on socket");
        close(server.listenfd);
        return -1;
    }
    return 0;
}

// 处理客户端请求
void *rpc_handle_client(void *arg) {
    RpcClient *client = (RpcClient *)arg;
    ssize_t bytes_read;
    ssize_t bytes_write;

    while(1) {
        // 读取客户端请求, 先读取函数ID
        ReqHeader header;
        bytes_read = read(client->sockfd, &header, sizeof(header));
        if(bytes_read <= 0) {
            if(bytes_read == 0) {
                printf("Client disconnected: %d\n", client->sockfd);
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
        client->iov_read_count = 0;
        client->iov_read2_count = 0;
        client->iov_send_count = 0;
        client->iov_send2_count = 0;
        client->funcId = header.funcId;
        printf("client->funcId: %x\n", client->funcId);
        // 取得函数ID对应的处理函数
        RequestHandler handler = get_handler(client->funcId);
        if(handler == NULL) {
            fprintf(stderr, "No handler found for function ID: %x\n", client->funcId);
            break;
        }
        handler(client);
        client->iov_read_count = 0;
        client->iov_read2_count = 0;
        client->iov_send_count = 0;
        client->iov_send2_count = 0;
    }
DONE:
    for(auto it = client->tmp_server_bufers.begin(); it != client->tmp_server_bufers.end(); it++) {
        free(*it);
    }
    client->tmp_server_bufers.clear();
    shutdown(client->sockfd, SHUT_RDWR);
    close(client->sockfd);
    delete client;
    return NULL;
}

void *rpc_accept_clients(void *arg) {
    while(1) {
        int connfd = accept(server.listenfd, NULL, NULL);
        if(connfd < 0) {
            perror("Failed to accept connection");
            continue;
        }

        handshake_request handshake_req;
        handshake_response handshake_rsp;
        ssize_t bytes_read = read(connfd, &handshake_req, sizeof(handshake_req));
        if(bytes_read != sizeof(handshake_req)) {
            shutdown(connfd, SHUT_RDWR);
            close(connfd);
            continue;
        }

        if(handshake_req.version_key != server.version_key) {
            handshake_rsp.status = 1;
        } else {
            handshake_rsp.status = 0;
        }
        ssize_t bytes_write = write(connfd, &handshake_rsp, sizeof(handshake_rsp));
        if(bytes_write != sizeof(handshake_rsp)) {
            shutdown(connfd, SHUT_RDWR);
            close(connfd);
            continue;
        }

        RpcClient *client = new RpcClient();
        client->sockfd = connfd;
        uuid_copy(client->clientId, handshake_req.id);

        if(handshake_req.is_async) {
            char uuid_str[37]; // UUID 字符串长度为 36 字符 + 终止符
            uuid_unparse(handshake_req.id, uuid_str);
            std::string key = std::string(uuid_str);
            server.client_async_conn[key] = client;
            printf("New async client connected: %d\n", connfd);
            continue;
        }

        printf("New client connected: %d\n", connfd);
        pthread_t thread_id;
        if(pthread_create(&thread_id, NULL, rpc_handle_client, client) != 0) {
            perror("Failed to create thread");
            delete client;
            close(connfd);
            continue;
        }
        server.threads.push_back(thread_id);
    }
    return NULL;
}

// 关闭服务器
void rpc_close_server() {
    if(server.listenfd >= 0) {
        shutdown(server.listenfd, SHUT_RDWR);
        close(server.listenfd);
        server.listenfd = -1;
    }

    // Clean up async connections
    for(auto &pair : server.client_async_conn) {
        if(pair.second) {
            shutdown(pair.second->sockfd, SHUT_RDWR);
            close(pair.second->sockfd);
            delete pair.second;
        }
    }
    server.client_async_conn.clear();

    for(auto thread_id : server.threads) {
        pthread_join(thread_id, NULL);
    }
    server.threads.clear();
}

int main(int argc, char *argv[]) {
    signal(SIGPIPE, SIG_IGN);
    server.version_key = VERSION_KEY;
    if(rpc_bind(12345) < 0) {
        return EXIT_FAILURE;
    }
    printf("Server is running on port 12345...\n");

    // 启动服务器线程
    pthread_t server_thread;
    if(pthread_create(&server_thread, NULL, rpc_accept_clients, nullptr) != 0) {
        perror("Failed to start server thread");
        rpc_close_server();
        return EXIT_FAILURE;
    }

    // 等待服务器线程结束
    pthread_join(server_thread, NULL);

    // 关闭服务器
    rpc_close_server();
    return EXIT_SUCCESS;
}
