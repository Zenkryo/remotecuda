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
class RpcConn;
class RpcServer;

// 异常类
class RpcException : public std::runtime_error {
  public:
    explicit RpcException(const std::string &message, int line = 0) : std::runtime_error(message + (line > 0 ? " [line " + std::to_string(line) + "]" : "")) {}
};

// 内存异常类
class RpcMemoryException : public RpcException {
  public:
    explicit RpcMemoryException(const std::string &message, int line = 0) : RpcException(message, line) {}
};

// 错误码定义
enum class RpcError { OK = 0, CONNECTION_CLOSED = -1, WRITE_ERROR = -2, READ_ERROR = -3, CONNECT_TIMEOUT = -4, ACCEPT_TIMEOUT = -5, WRITE_TIMEOUT = -6, READ_TIMEOUT = -7, INVALID_SOCKET = -8, INVALID_ADDRESS = -9, HANDSHAKE_FAILED = -10, QUIT = -11 };

// 常量定义
constexpr size_t MAX_IOV_COUNT = 64;
constexpr size_t MAX_CONNECTIONS = 1;
constexpr uint16_t VERSION_KEY = 0x5544;

// 超时时间定义（毫秒）
constexpr int CONNECT_TIMEOUT_MS = 5000;  // 连接超时
constexpr int ACCEPT_TIMEOUT_MS = 1000;   // 接受连接超时
constexpr int WRITE_TIMEOUT_MS = 3000;    // 写操作超时
constexpr int READ_TIMEOUT_MS = 3000;     // 读操作超时
constexpr int SHUTDOWN_TIMEOUT_MS = 1000; // 关闭超时

// 握手请求和响应结构
struct HandshakeRequest {
    uuid_t id;
    bool is_async;
    uint16_t version_key;
} __attribute__((packed));

struct HandshakeResponse {
    int status;
} __attribute__((packed));

// RPC连接类
class RpcConn {
  public:
    RpcConn(uint16_t version_key, uuid_t client_id, bool is_server = false);
    ~RpcConn();

    // 禁用拷贝
    RpcConn(const RpcConn &) = delete;
    RpcConn &operator=(const RpcConn &) = delete;

    // 连接管理
    RpcError connect(const std::string &server, uint16_t port, bool is_async = false);
    RpcError reconnect();
    RpcError disconnect();
    bool is_connected() const;

    // 请求准备和发送
    void reset();
    void prepare_request(uint32_t func_id);
    void write(const void *data, size_t len, bool with_len = false);
    void read(void *buffer, size_t len, bool with_len = false);

    // 请求发送
    RpcError submit_request();

    // 响应处理
    RpcError prepare_response();
    RpcError submit_response();

    // 异步读取
    RpcError read_all_now(void **buffer, size_t *size, int count);
    RpcError read_one_now(void *buffer, size_t size, bool with_len = false);

    // 获取iov读写计数
    int get_iov_read_count(bool with_len) const;
    int get_iov_send_count(bool with_len) const;

    // 十六进制转储
    void hexdump(const char *desc, const void *buf, size_t len);

    // 友元声明
    friend class RpcServer;
    friend class RpcClient;

  private:
    std::atomic<int> sockfd_{-1}; // 使用原子类型
    uint32_t func_id_;
    uuid_t client_id_;
    std::string client_id_str_;
    uint16_t version_key_;
    bool is_server;
    std::atomic<bool> running_{false};

    std::string server_;
    uint16_t port_;
    bool is_async_;

    std::vector<iovec> iov_send_;
    std::vector<iovec> iov_send2_;
    std::vector<iovec> iov_read_;
    std::vector<iovec> iov_read2_;

    std::set<void *> tmp_buffers_;

    // 新增超时控制相关方法
    RpcError wait_for_readable(int timeout_ms);
    RpcError wait_for_writable(int timeout_ms);
    RpcError set_nonblocking();

    // 修改内部方法返回类型
    RpcError write_all(bool with_len);
    RpcError read_all(bool with_len);
    RpcError write_full_iovec(std::vector<iovec> &iov);
    RpcError read_full_iovec(std::vector<iovec> &iov);
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
    using RequestHandler = std::function<int(RpcConn *)>;
    void register_handler(uint32_t func_id, RequestHandler handler);

    // 获取异步连接
    RpcConn *get_async_conn(const std::string &client_id_str);

  private:
    int listenfd_;
    uint16_t version_key_;
    std::atomic<bool> running_{false};

    std::map<std::string, std::unique_ptr<RpcConn>> async_conns_;
    std::vector<std::shared_ptr<RpcConn>> sync_conns_; // 所有的同步连接
    std::vector<std::thread> worker_threads_;
    std::map<uint32_t, RequestHandler> handlers_;

    std::mutex async_mutex_;

    // 修改内部方法返回类型
    void accept_loop();
    void handle_request(std::shared_ptr<RpcConn> conn);
};

// RPC客户端类
class RpcClient {
  public:
    RpcClient(uint16_t version_key);
    ~RpcClient();

    // 禁用拷贝
    RpcClient(const RpcClient &) = delete;
    RpcClient &operator=(const RpcClient &) = delete;

    // 连接管理
    RpcError connect(const std::string &server, uint16_t port, size_t sync_pool_size = 4);
    void disconnect();

    // 同步连接池管理
    RpcConn *acquire_connection();
    void release_connection(RpcConn *conn);
    bool is_connected() const { return !sync_conns_.empty() && async_conn_ != nullptr; }

    // 异步请求处理
    using AsyncRequestHandler = std::function<void(RpcConn *)>;
    void register_async_handler(uint32_t func_id, AsyncRequestHandler handler);

  private:
    std::string server_addr_;
    uint16_t server_port_;
    uint16_t version_key_;
    uuid_t client_id_;
    std::atomic<bool> running_{false};

    // 同步连接池
    std::vector<std::unique_ptr<RpcConn>> sync_conns_;
    std::set<RpcConn *> available_conns_;
    std::mutex sync_mutex_;

    // 异步连接
    std::unique_ptr<RpcConn> async_conn_;
    std::thread async_thread_;
    std::map<uint32_t, AsyncRequestHandler> async_handlers_;

    // 修改内部方法返回类型
    void async_receive_loop();
};

} // namespace rpc
