#include <stdexcept>
#include <cstring>
#include <iostream>
#include <fcntl.h>
#include <sys/select.h>
#include <errno.h>
#include "rpc_core.h"
namespace rpc {

RpcClient::RpcClient(uint16_t version_key) : server_addr_(), server_port_(0), version_key_(version_key) { uuid_generate(client_id_); }

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
        auto conn = std::unique_ptr<RpcConn>(new RpcConn(version_key_, client_id_));
        RpcError err = conn->connect(server, port);
        if(err != RpcError::OK) {
            return err;
        }
        available_sync_conns_.push_back(std::move(conn));
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

    // 清理同步连接池
    std::lock_guard<std::mutex> lock(sync_mutex_);
    for(auto &conn : using_sync_conns_) {
        if(conn) {
            conn->disconnect();
        }
    }
    using_sync_conns_.clear();
    for(auto &conn : available_sync_conns_) {
        if(conn) {
            conn->disconnect();
        }
    }
    available_sync_conns_.clear();

    // 清理异步连接
    if(async_conn_) {
        async_conn_->disconnect();
    }

    // 等待异步线程结束
    if(async_thread_.joinable()) {
        async_thread_.join();
    }

    // 最后释放异步连接
    async_conn_.reset();
}

RpcConn *RpcClient::acquire_connection() {
    std::lock_guard<std::mutex> lock(sync_mutex_);

    // 从可用连接中选择一个
    if(!available_sync_conns_.empty()) {
        auto conn = std::move(available_sync_conns_.back());
        RpcConn *conn_ptr = conn.get();
        available_sync_conns_.pop_back();
        using_sync_conns_.push_back(std::move(conn));
        return conn_ptr;
    }

    // 如果没有可用连接，创建新的连接
    auto conn = std::unique_ptr<RpcConn>(new RpcConn(version_key_, client_id_));
    RpcError err = conn->connect(server_addr_, server_port_);
    if(err != RpcError::OK) {
        return nullptr;
    }

    RpcConn *conn_ptr = conn.get();
    using_sync_conns_.push_back(std::move(conn));
    return conn_ptr;
}

void RpcClient::release_connection(RpcConn *conn) {
    if(!conn)
        return;

    std::lock_guard<std::mutex> lock(sync_mutex_);
    auto it = std::find_if(using_sync_conns_.begin(), using_sync_conns_.end(), [conn](const std::unique_ptr<RpcConn> &c) { return c.get() == conn; });
    if(it != using_sync_conns_.end()) {
        available_sync_conns_.push_back(std::move(*it));
        using_sync_conns_.erase(it);
    }
}

void RpcClient::register_async_handler(uint32_t func_id, AsyncRequestHandler handler) { async_handlers_[func_id] = handler; }

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
        AsyncRequestHandler handler;

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
