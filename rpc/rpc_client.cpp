#include <stdexcept>
#include <cstring>
#include <iostream>
#include <fcntl.h>
#include <sys/select.h>
#include <errno.h>
#include "rpc_core.h"
namespace rpc {

RpcClient::RpcClient(uint16_t version_key) : version_key_(version_key) { uuid_generate(client_id_); }

RpcClient::~RpcClient() { disconnect(); }

RpcError RpcClient::connect(const std::string &server, uint16_t port, size_t sync_pool_size) {
    if(running_) {
        return RpcError::INVALID_SOCKET;
    }
    running_ = true;

    server_addr_ = server;
    server_port_ = port;
    std::lock_guard<std::mutex> lock(sync_mutex_);

    // 创建同步连接池
    for(size_t i = 0; i < sync_pool_size; i++) {
        auto conn = std::unique_ptr<RpcConn>(new RpcConn(version_key_, client_id_));
        RpcError err = conn->connect(server, port);
        if(err != RpcError::OK) {
            return err;
        }
        sync_conns_.push_back(std::move(conn));
    }

    // 创建异步连接
    async_conn_ = std::unique_ptr<RpcConn>(new RpcConn(version_key_, client_id_));
    RpcError err = async_conn_->connect(server, port, true);
    if(err != RpcError::OK) {
        return err;
    }

    // 启动异步接收线程
    async_thread_ = std::thread(&RpcClient::async_receive_loop, this);

    return RpcError::OK;
}

void RpcClient::disconnect() {
    if(!running_) {
        return;
    }

    running_ = false;

    std::lock_guard<std::mutex> lock(sync_mutex_);

    // 释放所有同步连接
    sync_conns_.clear();

    // 清理异步连接
    if(async_conn_) {
        async_conn_->disconnect();
    }

    // 等待异步线程结束
    if(async_thread_.joinable()) {
        async_thread_.join();
    }

    // 释放异步连接
    async_conn_.reset();
}

bool RpcClient::is_connected() {
    std::lock_guard<std::mutex> lock(sync_mutex_);
    bool has_connected_sync_conn = false;
    for(auto &conn : sync_conns_) {
        if(conn->is_connected()) {
            has_connected_sync_conn = true;
            break;
        }
    }
    if(has_connected_sync_conn) {
        if(async_conn_ && async_conn_->is_connected()) {
            return true;
        }
    }
    return false;
}

RpcConn *RpcClient::acquire_connection() {
    std::lock_guard<std::mutex> lock(sync_mutex_);

    for(auto &conn : sync_conns_) {
        if(conn->is_connected() && !conn->is_using_) {
            conn->is_using_ = true;
            return conn.get();
        }
    }

    // 如果没有可用连接，创建新的连接
    auto conn = std::unique_ptr<RpcConn>(new RpcConn(version_key_, client_id_));
    RpcError err = conn->connect(server_addr_, server_port_);
    if(err != RpcError::OK) {
        return nullptr;
    }

    RpcConn *conn_ptr = conn.get();
    conn->is_using_ = true;
    sync_conns_.push_back(std::move(conn));
    return conn_ptr;
}

void RpcClient::release_connection(RpcConn *conn, bool to_close) {
    if(!conn)
        return;

    std::lock_guard<std::mutex> lock(sync_mutex_);
    auto it = std::find_if(sync_conns_.begin(), sync_conns_.end(), [conn](const std::unique_ptr<RpcConn> &c) { return c.get() == conn; });
    if(it != sync_conns_.end()) {
        if(to_close) {
            sync_conns_.erase(it);
        }
    }
    conn->is_using_ = false;
    conn->free_all_iov_buffers();
    if(to_close) {
        conn->disconnect();
    }
}

void RpcClient::register_async_handler(uint32_t func_id, RequestHandler handler) { async_handlers_[func_id] = handler; }

void RpcClient::async_receive_loop() {
    while(running_) {
        uint32_t func_id;
        RpcError err = async_conn_->read_one_now(&func_id, sizeof(func_id));
        if(err != RpcError::OK) {
            if(async_conn_->reconnect() != RpcError::OK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
            continue;
        }

        // 获取处理函数
        RequestHandler handler;

        auto it = async_handlers_.find(func_id);
        if(it == async_handlers_.end()) {
            throw RpcException("Invalid function id", __LINE__);
        }
        handler = it->second;

        // 处理请求
        handler(async_conn_.get());
    }
}

} // namespace rpc
