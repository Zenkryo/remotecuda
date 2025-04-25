#include "rpc_core.h"
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <fcntl.h>
#include <sys/select.h>
#include <errno.h>

namespace rpc {

RpcServer::RpcServer(uint16_t port, uint16_t version_key) : listenfd_(-1), version_key_(version_key) {
    // 创建监听socket
    listenfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if(listenfd_ < 0) {
        throw RpcException("Failed to create socket: " + std::string(strerror(errno)), __LINE__);
    }

    // 设置为非阻塞模式
    int flags = fcntl(listenfd_, F_GETFL, 0);
    if(flags == -1) {
        close(listenfd_);
        throw RpcException("Failed to get socket flags: " + std::string(strerror(errno)), __LINE__);
    }
    if(fcntl(listenfd_, F_SETFL, flags | O_NONBLOCK) == -1) {
        close(listenfd_);
        throw RpcException("Failed to set non-blocking mode: " + std::string(strerror(errno)), __LINE__);
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
        return;
    }
    running_ = true;
    accept_loop();
}

void RpcServer::stop() {
    if(!running_) {
        return;
    }
    running_ = false;

    // 关闭监听socket
    if(listenfd_ >= 0) {
        shutdown(listenfd_, SHUT_RDWR);
        close(listenfd_);
        listenfd_ = -1;
    }

    // 清理异步连接
    std::lock_guard<std::mutex> lock(async_mutex_);
    for(auto &pair : async_conns_) {
        if(pair.second) {
            pair.second->disconnect();
        }
    }
    async_conns_.clear();

    // 清理同步连接
    for(auto &conn : sync_conns_) {
        if(conn) {
            conn->disconnect();
        }
    }
    sync_conns_.clear();

    // 等待所有工作线程结束
    for(auto &thread : worker_threads_) {
        if(thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

void RpcServer::register_handler(uint32_t func_id, RequestHandler handler) { handlers_[func_id] = handler; }

RpcConn *RpcServer::get_async_conn(const std::string &client_id_str) {
    std::lock_guard<std::mutex> lock(async_mutex_);
    auto it = async_conns_.find(client_id_str);
    return (it != async_conns_.end()) ? it->second.get() : nullptr;
}

void RpcServer::accept_loop() {
    while(running_) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(listenfd_, &read_fds);

        struct timeval tv;
        tv.tv_sec = ACCEPT_TIMEOUT_MS / 1000;
        tv.tv_usec = (ACCEPT_TIMEOUT_MS % 1000) * 1000;

        int ret = select(listenfd_ + 1, &read_fds, nullptr, nullptr, &tv);
        if(ret <= 0) {
            continue;
        }

        int connfd = accept(listenfd_, NULL, NULL);
        if(connfd < 0) {
            continue;
        }

        // 设置为非阻塞模式
        int flags = fcntl(connfd, F_GETFL, 0);
        if(flags == -1) {
            close(connfd);
            continue;
        }
        if(fcntl(connfd, F_SETFL, flags | O_NONBLOCK) == -1) {
            close(connfd);
            continue;
        }

        // 处理握手
        HandshakeRequest handshake_req;
        FD_ZERO(&read_fds);
        FD_SET(connfd, &read_fds);

        tv.tv_sec = READ_TIMEOUT_MS / 1000;
        tv.tv_usec = (READ_TIMEOUT_MS % 1000) * 1000;

        ret = select(connfd + 1, &read_fds, nullptr, nullptr, &tv);
        if(ret <= 0) {
            close(connfd);
            continue;
        }

        if(read(connfd, &handshake_req, sizeof(handshake_req)) != sizeof(handshake_req)) {
            close(connfd);
            continue;
        }

        HandshakeResponse handshake_rsp;
        handshake_rsp.status = (handshake_req.version_key == version_key_) ? 0 : 1;

        fd_set write_fds;
        FD_ZERO(&write_fds);
        FD_SET(connfd, &write_fds);

        tv.tv_sec = WRITE_TIMEOUT_MS / 1000;
        tv.tv_usec = (WRITE_TIMEOUT_MS % 1000) * 1000;

        ret = select(connfd + 1, nullptr, &write_fds, nullptr, &tv);
        if(ret <= 0) {
            close(connfd);
            continue;
        }

        if(write(connfd, &handshake_rsp, sizeof(handshake_rsp)) != sizeof(handshake_rsp)) {
            close(connfd);
            continue;
        }

        if(handshake_rsp.status != 0) {
            close(connfd);
            continue;
        }

        // 创建新的客户端连接
        auto client = std::make_unique<RpcConn>(version_key_, handshake_req.id, true);
        client->sockfd_ = connfd;
        client->running_ = true;

        if(handshake_req.is_async) {
            // 处理异步连接, 服务器端保存客户端和异步连接的对应关系
            std::lock_guard<std::mutex> lock(async_mutex_);
            if(async_conns_.find(client->client_id_str_) != async_conns_.end()) {
                async_conns_.erase(client->client_id_str_);
            }
            async_conns_[client->client_id_str_] = std::move(client);
        } else {
            // 处理同步连接, 创建工作线程处理请求
            std::shared_ptr<RpcConn> client_ptr = std::move(client);
            sync_conns_.push_back(client_ptr);

            worker_threads_.emplace_back(&RpcServer::handle_request, this, client_ptr);
        }
    }
}

void RpcServer::handle_request(std::shared_ptr<RpcConn> conn) {
    while(running_) {
        uint32_t func_id = 0;
        RpcError err = conn->read_one_now(&func_id, sizeof(func_id));
        if(err != RpcError::OK) {
            break;
        }

        // 获取处理函数
        RequestHandler handler;
        auto it = handlers_.find(func_id);
        if(it == handlers_.end()) {
            break;
        }
        handler = it->second;

        // 处理请求
        if(handler(conn.get()) != 0) {
            break;
        }
    }
    conn->disconnect();
}

} // namespace rpc
