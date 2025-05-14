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
#include <algorithm>

namespace rpc {

constexpr uint16_t SERVER_PORT = 12345;

// 前向声明
class RpcConn;
class RpcServer;
class RpcBuffers;

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

// 请求处理函数类型
using RequestHandler = std::function<int(RpcConn *)>;

// 错误码定义
enum class RpcError {
    OK = 0,                 // 正确
    CONNECTION_CLOSED = -1, // 连接关闭
    WRITE_ERROR = -2,       // 写错误
    READ_ERROR = -3,        // 读错误
    CONNECT_TIMEOUT = -4,   // 连接超时
    ACCEPT_TIMEOUT = -5,    // 接受连接超时
    WRITE_TIMEOUT = -6,     // 写操作超时
    READ_TIMEOUT = -7,      // 读操作超时
    INVALID_SOCKET = -8,    // 无效的socket
    INVALID_ADDRESS = -9,   // 无效的地址
    HANDSHAKE_FAILED = -10, // 握手失败
    MALLOC_FAILED = -11,    // 内存分配失败
    QUIT = -12              // 退出
};

// 超时时间定义（毫秒）
constexpr int CONNECT_TIMEOUT_MS = 5000;  // 连接超时
constexpr int ACCEPT_TIMEOUT_MS = 1000;   // 接受连接超时
constexpr int WRITE_TIMEOUT_MS = 3000;    // 写操作超时
constexpr int READ_TIMEOUT_MS = 3000;     // 读操作超时
constexpr int SHUTDOWN_TIMEOUT_MS = 1000; // 关闭超时

// 握手请求和响应结构
struct HandshakeRequest {
    uuid_t id;            // 客户端id
    bool is_async;        // 是否适用于异步消息的连接
    uint16_t version_key; // 版本号
} __attribute__((packed));

struct HandshakeResponse {
    int status; // 成功还是失败
} __attribute__((packed));

// RPC连接类，代表一条用于RPC通讯的tcp连接，用于客户端，也用于服务端
class RpcConn {
  public:
    RpcConn(uint16_t version_key, uuid_t client_id, bool is_server = false);
    ~RpcConn();

    // 禁用拷贝
    RpcConn(const RpcConn &) = delete;
    RpcConn &operator=(const RpcConn &) = delete;

    // 连接
    RpcError connect(const std::string &server, uint16_t port, bool is_async = false);

    // 断线重连
    RpcError reconnect();

    // 断开连接
    RpcError disconnect();

    // 是否连接
    bool is_connected() const { return sockfd_ >= 0; }

    // 请求准备
    void prepare_request(uint32_t func_id);

    // 请求发送
    RpcError submit_request();

    // 响应准备
    RpcError prepare_response();

    // 响应发送
    RpcError submit_response();

    // 准备写入
    void write(const void *data, size_t len, bool with_len = false);

    // 准备读取
    void read(void *buffer, size_t len, bool with_len = false);

    // 立即读取多个带长度的数据
    RpcError read_all_now(void **buffer, size_t *size, int count);

    // 立即读取一个数据
    RpcError read_one_now(void *buffer, size_t size, bool with_len = false);

    // 获取iov读计数
    int get_iov_read_count(bool with_len) const { return with_len ? iov_read2_.size() : iov_read_.size(); }

    // 获取iov写计数
    int get_iov_send_count(bool with_len) const { return with_len ? iov_send2_.size() : iov_send_.size(); }

    // 十六进制转储
    void hexdump(const char *desc, const void *buf, size_t len);

    // 获取主机缓冲区
    void *alloc_host_buffer(size_t size);

    // 释放主机缓冲区
    void free_host_buffer(void *ptr);

    // 获取iov临时缓冲区
    void *alloc_iov_buffer(size_t size);

    // 释放iov临时缓冲区
    void free_iov_buffer(void *ptr);

    // 释放所有iov临时缓冲区
    void free_all_iov_buffers();

    // 获取client id
    std::string get_client_id() const { return client_id_str_; }

    // 友元声明
    friend class RpcServer;
    friend class RpcClient;

  private:
    std::atomic<int> sockfd_{-1};                       // 使用原子类型
    std::atomic<bool> running_{false};                  // 是否运行, 用于通知退出
    std::atomic<bool> is_using_{false};                 // 是否正在使用，防止并行使用
    uuid_t client_id_;                                  // 客户端id
    uint16_t version_key_;                              // 版本号
    std::string client_id_str_;                         // 客户端id字符串
    bool is_server;                                     // 是否是用于服务端
    bool is_async_;                                     // 是否是异步连接
    std::set<void *> iov_buffers_;                      // 临时iov数据的缓冲区
    std::mutex iov_buffers_mutex_;                      // 临时iov缓冲区互斥锁
    std::string server_addr_;                           // 服务器地址
    uint16_t server_port_;                              // 服务器端口
    uint32_t func_id_;                                  // 请求的函数id
    std::vector<iovec> iov_send_;                       // iov发送缓冲区, 用于发送不带长度的数据
    std::vector<iovec> iov_send2_;                      // iov发送缓冲区, 用于发送带长度的数据
    std::vector<iovec> iov_read_;                       // iov读取缓冲区, 用于读取不带长度的数据
    std::vector<iovec> iov_read2_;                      // iov读取缓冲区, 用于读取带长度的数据
    RpcError wait_for_readable(int timeout_ms);         // 等待可读
    RpcError wait_for_writable(int timeout_ms);         // 等待可写
    RpcError set_nonblocking();                         // 设置非阻塞
    RpcError write_all(bool with_len);                  // 发送所有iov发送缓冲区
    RpcError read_all(bool with_len);                   // 读取所有iov读取缓冲区
    RpcError write_full_iovec(std::vector<iovec> &iov); // 完整发送一个iov
    RpcError read_full_iovec(std::vector<iovec> &iov);  // 完整读取一个iov
};

// RPC服务器类, 代表一个RPC服务器
class RpcServer {
  public:
    // 获取单例实例
    static RpcServer &getInstance() {
        static RpcServer instance(SERVER_PORT);
        return instance;
    }

    // 禁用拷贝
    RpcServer(const RpcServer &) = delete;
    RpcServer &operator=(const RpcServer &) = delete;

    // 启动服务
    void start(uint16_t version_key);

    // 停止服务
    void stop();

    // 注册处理函数
    void register_handler(uint32_t func_id, RequestHandler handler);

    // 获取异步连接
    RpcConn *get_async_conn(const std::string &client_id_str);

    // 释放异步连接
    void release_async_conn(RpcConn *conn, bool to_close = false);

  private:
    // 私有构造函数和析构函数
    RpcServer(uint16_t port);
    ~RpcServer();
    int listenfd_;                                                // 监听socket
    uint16_t version_key_;                                        // 版本号
    std::atomic<bool> running_{false};                            // 是否运行, 用于通知退出
    std::map<std::string, std::unique_ptr<RpcConn>> async_conns_; // 每个client id一个异步连接
    std::mutex async_mutex_;                                      // 异步连接互斥锁
    std::vector<std::thread> worker_threads_;                     // 工作线程
    std::map<uint32_t, RequestHandler> handlers_;                 // 每个函数id一个处理函数
    void handle_request(std::shared_ptr<RpcConn> conn);           // 处理请求
    void accept_loop();                                           // 接受连接循环
};

// RPC客户端类, 代表一个RPC客户端
class RpcClient {
  public:
    RpcClient(uint16_t version_key);
    ~RpcClient();

    // 禁用拷贝
    RpcClient(const RpcClient &) = delete;
    RpcClient &operator=(const RpcClient &) = delete;

    // 建立和服务器端的多个同步连接和一个异步连接
    RpcError connect(const std::string &server, uint16_t port, size_t sync_pool_size = 4);

    // 断开和服务器端的所有连接
    void disconnect();

    // 获取和服务器端的一个同步连接
    RpcConn *acquire_connection();

    // 释放和服务器端的一个同步连接
    void release_connection(RpcConn *conn, bool to_close = false);

    // 是否连接
    bool is_connected();

    // 注册异步请求处理函数
    void register_async_handler(uint32_t func_id, RequestHandler handler);

  private:
    std::string server_addr_;                           // 服务器地址
    uint16_t server_port_;                              // 服务器端口
    uuid_t client_id_;                                  // 客户端id
    uint16_t version_key_;                              // 版本号
    std::atomic<bool> running_{false};                  // 是否运行, 用于通知退出
    std::vector<std::unique_ptr<RpcConn>> sync_conns_;  // 同步连接池
    std::mutex sync_mutex_;                             // 同步连接池互斥锁
    std::unique_ptr<RpcConn> async_conn_;               // 异步连接
    std::thread async_thread_;                          // 异步消息处理线程
    std::map<uint32_t, RequestHandler> async_handlers_; // 异步请求处理函数
    void async_receive_loop();                          // 异步接收循环
};

// RPC缓冲区管理类, 用于管理RPC缓冲区
class RpcBuffers {
  public:
    // 获取单例实例
    static RpcBuffers &getInstance() {
        static RpcBuffers instance;
        return instance;
    }

    // 禁用拷贝和赋值
    RpcBuffers(const RpcBuffers &) = delete;
    RpcBuffers &operator=(const RpcBuffers &) = delete;

    // 分配主机内存
    void *alloc_rpc_buffer(std::string client_id, size_t size);

    // 释放主机内存
    void free_rpc_buffer(std::string client_id, void *ptr);

    // 增加连接计数
    void increment_connection_count(const std::string &client_id);

    // 减少连接计数
    void decrement_connection_count(const std::string &client_id);

  private:
    // 私有构造函数和析构函数
    RpcBuffers() = default;
    ~RpcBuffers() = default;

    std::map<std::string, std::set<void *>> buffers_;           // 每个客户端的所有rpc缓冲区
    std::map<std::string, std::atomic<int>> connection_counts_; // 每个客户端的连接计数
    std::mutex buffers_mutex_;                                  // 缓冲区互斥锁
};
} // namespace rpc
