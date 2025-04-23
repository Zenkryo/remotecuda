#include "rpc_core.h"
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace rpc {

RpcServer::RpcServer(uint16_t port, uint16_t version_key) : listenfd_(-1), version_key_(version_key), running_(false) {

    // 创建监听socket
    listenfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if(listenfd_ < 0) {
        throw RpcException("Failed to create socket: " + std::string(strerror(errno)), __LINE__);
    }

    // 设置SO_REUSEADDR选项
    int reuse = 1;
    if(setsockopt(listenfd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        close(listenfd_);
        throw RpcException("Failed to set SO_REUSEADDR: " + std::string(strerror(errno)), __LINE__);
    }

    // 绑定地址
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    if(bind(listenfd_, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        close(listenfd_);
        throw RpcException("Failed to bind: " + std::string(strerror(errno)), __LINE__);
    }

    // 开始监听
    if(listen(listenfd_, 5) < 0) {
        close(listenfd_);
        throw RpcException("Failed to listen: " + std::string(strerror(errno)), __LINE__);
    }
}

RpcServer::~RpcServer() { stop(); }

void RpcServer::start() {
    if(running_) {
        throw RpcException("Server already running", __LINE__);
    }

    running_ = true;
    accept_loop();
}

void RpcServer::stop() {
    if(!running_)
        return;

    running_ = false;

    // 关闭监听socket
    if(listenfd_ >= 0) {
        shutdown(listenfd_, SHUT_RDWR);
        close(listenfd_);
        listenfd_ = -1;
    }

    // 清理异步连接
    std::lock_guard<std::mutex> lock(mutex_);
    for(auto &pair : async_conns_) {
        if(pair.second) {
            pair.second->disconnect();
        }
    }
    async_conns_.clear();

    // 等待所有工作线程结束
    for(auto &thread : worker_threads_) {
        if(thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

void RpcServer::register_handler(uint32_t func_id, RequestHandler handler) {
    std::lock_guard<std::mutex> lock(mutex_);
    handlers_[func_id] = handler;
}

void RpcServer::accept_loop() {
    while(running_) {
        int connfd = accept(listenfd_, NULL, NULL);
        if(connfd < 0) {
            if(errno == EINTR)
                continue;
            throw RpcException("Failed to accept: " + std::string(strerror(errno)), __LINE__);
        }

        // 处理握手
        HandshakeRequest handshake_req;
        if(read(connfd, &handshake_req, sizeof(handshake_req)) != sizeof(handshake_req)) {
            close(connfd);
            continue;
        }

        HandshakeResponse handshake_rsp;
        handshake_rsp.status = (handshake_req.version_key == version_key_) ? 0 : 1;

        if(write(connfd, &handshake_rsp, sizeof(handshake_rsp)) != sizeof(handshake_rsp)) {
            close(connfd);
            continue;
        }

        if(handshake_rsp.status != 0) {
            close(connfd);
            continue;
        }

        // 创建新的客户端连接
        auto client = std::make_unique<RpcConn>();
        client->is_server = true;
        client->sockfd_ = connfd;
        uuid_copy(client->client_id_, handshake_req.id);

        if(handshake_req.is_async) {
            // 处理异步连接
            char uuid_str[37];
            uuid_unparse(handshake_req.id, uuid_str);
            std::string key = uuid_str;

            std::lock_guard<std::mutex> lock(mutex_);
            async_conns_[key] = std::move(client);
        } else {
            // 处理同步连接
            worker_threads_.emplace_back(&RpcServer::handle_request, this, std::move(client));
        }
    }
}

void RpcServer::handle_request(std::unique_ptr<RpcConn> conn) {
    while(running_) {
        uint32_t func_id;

        conn->read_one_now(&func_id, sizeof(func_id));
        // 获取处理函数
        RequestHandler handler;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = handlers_.find(func_id);
            if(it == handlers_.end()) {
                break;
            }
            handler = it->second;
        }
        // 处理请求
        if(handler(conn.get()) != 0) {
            break;
        }
    }
    conn->disconnect();
}

} // namespace rpc
