#include "rpc_core.h"
#include <stdexcept>
#include <chrono>
#include <thread>

namespace rpc {

RpcClient::RpcClient() : running_(false) {}

RpcClient::~RpcClient() { disconnect(); }

void RpcClient::connect(const std::string &server, uint16_t port, size_t sync_pool_size) {
    if(is_connected()) {
        throw RpcException("Client is already connected");
    }

    server_addr_ = server;
    server_port_ = port;
    running_ = true;

    // 创建同步连接池
    {
        std::lock_guard<std::mutex> lock(sync_mutex_);
        sync_conns_.reserve(sync_pool_size);
        for(size_t i = 0; i < sync_pool_size; ++i) {
            auto conn = std::make_unique<RpcConn>();
            conn->connect(server, port, false);
            available_conns_.insert(conn.get());
            sync_conns_.push_back(std::move(conn));
        }
    }

    // 创建异步连接
    async_conn_ = std::make_unique<RpcConn>();
    async_conn_->connect(server, port, true);

    // 启动异步接收线程
    async_thread_ = std::thread(&RpcClient::async_receive_loop, this);
}

void RpcClient::disconnect() {
    if(!is_connected()) {
        return;
    }

    running_ = false;

    // 等待异步线程结束
    if(async_thread_.joinable()) {
        async_thread_.join();
    }

    // 关闭所有连接
    {
        std::lock_guard<std::mutex> lock(sync_mutex_);
        sync_conns_.clear();
        available_conns_.clear();
    }

    async_conn_.reset();
}

RpcConn *RpcClient::acquire_connection() {
    std::lock_guard<std::mutex> lock(sync_mutex_);

    if(available_conns_.empty()) {
        throw RpcException("No available connections in the pool");
    }

    auto conn = *available_conns_.begin();
    available_conns_.erase(available_conns_.begin());
    return conn;
}

void RpcClient::release_connection(RpcConn *conn) {
    if(!conn) {
        return;
    }

    std::lock_guard<std::mutex> lock(sync_mutex_);
    available_conns_.insert(conn);
}

void RpcClient::register_async_handler(uint32_t func_id, AsyncRequestHandler handler) {
    std::lock_guard<std::mutex> lock(async_mutex_);
    async_handlers_[func_id] = std::move(handler);
}

void RpcClient::async_receive_loop() {
    while(running_) {
        if(!async_conn_->is_connected()) {
            try {
                async_conn_->connect(server_addr_, server_port_, true);
            } catch(const std::exception &e) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
        }

        try {
            handle_async_request(async_conn_.get());
        } catch(const RpcConnException &e) {
            async_conn_->disconnect();
        } catch(const std::exception &e) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

void RpcClient::handle_async_request(RpcConn *conn) {
    conn->reset();
    uint32_t func_id;
    conn->read_one_now(&func_id, sizeof(func_id));

    std::lock_guard<std::mutex> lock(async_mutex_);
    auto it = async_handlers_.find(func_id);
    if(it != async_handlers_.end()) {
        it->second(conn);
    } else {
        throw RpcException("No handler registered for async function ID: " + std::to_string(func_id));
    }
}

} // namespace rpc
