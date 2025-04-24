#include "rpc_core.h"
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <fcntl.h>
#include <sys/select.h>
#include <errno.h>

namespace rpc {

RpcClient::RpcClient(uint16_t version_key) : server_addr_(), server_port_(0), version_key_(version_key), running_(false) { uuid_generate(client_id_); }

RpcClient::~RpcClient() { disconnect(); }

RpcError RpcClient::connect(const std::string &server, uint16_t port, size_t sync_pool_size) {
    if(running_) {
        return RpcError::INVALID_SOCKET;
    }

    server_addr_ = server;
    server_port_ = port;
    running_ = true;

    // 创建同步连接池
    for(size_t i = 0; i < sync_pool_size; i++) {
        auto conn = std::make_unique<RpcConn>(version_key_, client_id_);
        RpcError err = conn->connect(server, port);
        if(err != RpcError::OK) {
            return err;
        }
        sync_conns_.push_back(std::move(conn));
    }

    // 创建异步连接
    async_conn_ = std::make_unique<RpcConn>(version_key_, client_id_);
    RpcError err = async_conn_->connect(server, port, true);
    if(err != RpcError::OK) {
        return err;
    }

    // 启动异步接收线程
    async_thread_ = std::thread(&RpcClient::async_receive_loop, this);

    return RpcError::OK;
}

void RpcClient::disconnect() {
    if(!running_)
        return;

    running_ = false;

    // 清理同步连接池
    std::lock_guard<std::mutex> lock(sync_mutex_);
    for(auto &conn : sync_conns_) {
        if(conn) {
            conn->disconnect();
        }
    }
    sync_conns_.clear();

    // 清理异步连接
    if(async_conn_) {
        async_conn_->disconnect();
        async_conn_.reset();
    }

    // 等待异步线程结束
    if(async_thread_.joinable()) {
        async_thread_.join();
    }

    available_conns_.clear();
}

RpcConn *RpcClient::acquire_connection() {
    std::lock_guard<std::mutex> lock(sync_mutex_);
    if(sync_conns_.empty()) {
        return nullptr;
    }

    // 从可用连接中选择一个
    if(!available_conns_.empty()) {
        auto it = available_conns_.begin();
        RpcConn *conn = *it;
        available_conns_.erase(it);
        return conn;
    }

    // 如果没有可用连接，创建新的连接
    auto conn = std::make_unique<RpcConn>(version_key_, client_id_);
    RpcError err = conn->connect(server_addr_, server_port_);
    if(err != RpcError::OK) {
        return nullptr;
    }

    RpcConn *conn_ptr = conn.get();
    sync_conns_.push_back(std::move(conn));
    return conn_ptr;
}

void RpcClient::release_connection(RpcConn *conn) {
    if(!conn)
        return;

    std::lock_guard<std::mutex> lock(sync_mutex_);
    available_conns_.insert(conn);
}

void RpcClient::register_async_handler(uint32_t func_id, AsyncRequestHandler handler) { async_handlers_[func_id] = handler; }

RpcError RpcClient::async_receive_loop() {
    while(running_) {
        uint32_t func_id;
        RpcError err = async_conn_->read_one_now(&func_id, sizeof(func_id));
        if(err != RpcError::OK) {
            if(err == RpcError::CONNECTION_CLOSED) {
                break;
            }
            continue;
        }

        // 获取处理函数
        AsyncRequestHandler handler;
        {
            auto it = async_handlers_.find(func_id);
            if(it == async_handlers_.end()) {
                continue;
            }
            handler = it->second;
        }

        // 处理请求
        handler(async_conn_.get());
    }
    return RpcError::OK;
}

RpcError RpcClient::handle_async_request(RpcConn *conn) {
    if(!conn)
        return RpcError::INVALID_SOCKET;

    uint32_t func_id;
    RpcError err = conn->read_one_now(&func_id, sizeof(func_id));
    if(err != RpcError::OK) {
        return err;
    }

    // 获取处理函数
    AsyncRequestHandler handler;
    {
        auto it = async_handlers_.find(func_id);
        if(it == async_handlers_.end()) {
            return RpcError::INVALID_SOCKET;
        }
        handler = it->second;
    }

    // 处理请求
    handler(conn);
    return RpcError::OK;
}

} // namespace rpc
