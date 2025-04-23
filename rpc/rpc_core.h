#pragma once

#include <memory>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <functional>
#include <mutex>
#include <thread>
#include <atomic>
#include <system_error>
#include <uuid/uuid.h>
#include <sys/uio.h>
#include <arpa/inet.h>
#include <unistd.h>

namespace rpc {

// 前向声明
class RpcClient;
class RpcServer;

// 异常类
class RpcException : public std::runtime_error {
  public:
    explicit RpcException(const std::string &message) : std::runtime_error(message) {}
};

// 常量定义
constexpr size_t MAX_IOV_COUNT = 64;
constexpr size_t MAX_CONNECTIONS = 1;

// 握手请求和响应结构
struct HandshakeRequest {
    uuid_t id;
    bool is_async;
    uint16_t version_key;
} __attribute__((packed));

struct HandshakeResponse {
    int status;
} __attribute__((packed));

// RPC客户端类
class RpcClient {
  public:
    RpcClient();
    ~RpcClient();

    // 禁用拷贝
    RpcClient(const RpcClient &) = delete;
    RpcClient &operator=(const RpcClient &) = delete;

    // 连接管理
    void connect(const std::string &server, uint16_t port, bool is_async = false);
    void disconnect();
    bool is_connected() const { return sockfd_ >= 0; }

    // 请求准备和发送
    void prepare_request(uint32_t func_id);
    void write(const void *data, size_t len, bool with_len = false);
    void read(void *buffer, size_t len, bool with_len = false);
    int submit_request();

    // 响应处理
    int prepare_response();
    int submit_response();

    // 异步读取
    ssize_t read_all_now(void **buffer, size_t *size, int count);
    ssize_t read_one_now(void *buffer, size_t size, bool with_len = false);

    void hexdump(const char *desc, const void *buf, size_t len);

    // 友元声明
    friend class RpcServer;

  private:
    int sockfd_;
    uint32_t func_id_;
    uuid_t client_id_;
    bool is_server;

    std::vector<iovec> iov_send_;
    std::vector<iovec> iov_send2_;
    std::vector<iovec> iov_read_;
    std::vector<iovec> iov_read2_;

    std::set<void *> tmp_buffers_;
    std::mutex mutex_;
    std::atomic<bool> in_use_;

    // 内部辅助方法
    ssize_t write_full_iovec(std::vector<iovec> &iov);
    ssize_t read_full_iovec(std::vector<iovec> &iov);
    void cleanup_tmp_buffers();
};

// RPC服务器类
class RpcServer {
  public:
    RpcServer(uint16_t port, uint16_t version_key);
    ~RpcServer();

    // 禁用拷贝
    RpcServer(const RpcServer &) = delete;
    RpcServer &operator=(const RpcServer &) = delete;

    // 服务器控制
    void start();
    void stop();

    // 注册处理函数
    using RequestHandler = std::function<int(RpcClient *)>;
    void register_handler(uint32_t func_id, RequestHandler handler);

  private:
    int listenfd_;
    uint16_t version_key_;
    std::atomic<bool> running_;

    std::map<std::string, std::unique_ptr<RpcClient>> async_clients_;
    std::vector<std::thread> worker_threads_;
    std::map<uint32_t, RequestHandler> handlers_;

    std::mutex mutex_;

    // 内部方法
    void accept_loop();
    void handle_client(std::unique_ptr<RpcClient> client);
    void cleanup();
};

} // namespace rpc
