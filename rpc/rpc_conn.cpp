#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fcntl.h>
#include <sys/select.h>
#include <errno.h>
#include "rpc_core.h"

namespace rpc {

RpcConn::RpcConn(uint16_t version_key, uuid_t client_id, bool is_server) : func_id_(0), client_id_(), version_key_(version_key), is_server(is_server) {
    char uuid_str[37];
    uuid_copy(client_id_, client_id);
    uuid_unparse(client_id_, uuid_str);
    client_id_str_ = std::string(uuid_str);
    // 增加连接计数
    RpcBuffers::getInstance().increment_connection_count(client_id_str_);
}

RpcConn::~RpcConn() {
    // 减少连接计数
    RpcBuffers::getInstance().decrement_connection_count(client_id_str_);
}

RpcError RpcConn::set_nonblocking() {
    int flags = fcntl(sockfd_, F_GETFL, 0);
    if(flags == -1) {
        return RpcError::INVALID_SOCKET;
    }
    if(fcntl(sockfd_, F_SETFL, flags | O_NONBLOCK) == -1) {
        return RpcError::INVALID_SOCKET;
    }
    return RpcError::OK;
}

RpcError RpcConn::wait_for_readable(int timeout_ms) {
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(sockfd_, &read_fds);

    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    int ret = select(sockfd_ + 1, &read_fds, nullptr, nullptr, &tv);
    if(ret < 0) {
        return RpcError::READ_ERROR;
    } else if(ret == 0) {
        return RpcError::READ_TIMEOUT;
    }
    return RpcError::OK;
}

RpcError RpcConn::wait_for_writable(int timeout_ms) {
    fd_set write_fds;
    FD_ZERO(&write_fds);
    FD_SET(sockfd_, &write_fds);

    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    int ret = select(sockfd_ + 1, nullptr, &write_fds, nullptr, &tv);
    if(ret < 0) {
        return RpcError::WRITE_ERROR;
    } else if(ret == 0) {
        return RpcError::WRITE_TIMEOUT;
    }
    return RpcError::OK;
}

RpcError RpcConn::connect(const std::string &server, uint16_t port, bool is_async) {
    if(sockfd_ >= 0) {
        return RpcError::INVALID_SOCKET;
    }
    running_ = true;

    sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if(sockfd_ < 0) {
        return RpcError::INVALID_SOCKET;
    }

    // 设置为非阻塞模式
    RpcError err = set_nonblocking();
    if(err != RpcError::OK) {
        close(sockfd_);
        sockfd_ = -1;
        return err;
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    if(inet_pton(AF_INET, server.c_str(), &server_addr.sin_addr) <= 0) {
        close(sockfd_);
        sockfd_ = -1;
        return RpcError::INVALID_ADDRESS;
    }

    // 非阻塞连接
    int ret = ::connect(sockfd_, (struct sockaddr *)&server_addr, sizeof(server_addr));
    if(ret < 0) {
        if(errno == EINPROGRESS) {
            // 等待连接完成
            err = wait_for_writable(CONNECT_TIMEOUT_MS);
            if(err != RpcError::OK) {
                close(sockfd_);
                sockfd_ = -1;
                return err;
            }

            // 检查连接是否成功
            int error = 0;
            socklen_t len = sizeof(error);
            if(getsockopt(sockfd_, SOL_SOCKET, SO_ERROR, &error, &len) < 0 || error != 0) {
                close(sockfd_);
                sockfd_ = -1;
                return RpcError::CONNECTION_CLOSED;
            }
        } else {
            close(sockfd_);
            sockfd_ = -1;
            return RpcError::CONNECTION_CLOSED;
        }
    }

    // 发送握手请求
    HandshakeRequest handshake_req;
    uuid_copy(handshake_req.id, client_id_);
    handshake_req.is_async = is_async;
    handshake_req.version_key = version_key_;

    err = wait_for_writable(WRITE_TIMEOUT_MS);
    if(err != RpcError::OK) {
        close(sockfd_);
        sockfd_ = -1;
        return err;
    }

    if(::write(sockfd_, &handshake_req, sizeof(handshake_req)) != sizeof(handshake_req)) {
        close(sockfd_);
        sockfd_ = -1;
        return RpcError::WRITE_ERROR;
    }

    // 读取握手响应
    err = wait_for_readable(READ_TIMEOUT_MS);
    if(err != RpcError::OK) {
        close(sockfd_);
        sockfd_ = -1;
        return err;
    }

    HandshakeResponse handshake_rsp;
    if(::read(sockfd_, &handshake_rsp, sizeof(handshake_rsp)) != sizeof(handshake_rsp)) {
        close(sockfd_);
        sockfd_ = -1;
        return RpcError::READ_ERROR;
    }

    if(handshake_rsp.status != 0) {
        close(sockfd_);
        sockfd_ = -1;
        return RpcError::HANDSHAKE_FAILED;
    }

    // 保存连接信息，用于断线重连
    server_ = server;
    port_ = port;
    is_async_ = is_async;

    return RpcError::OK;
}

RpcError RpcConn::reconnect() {
    if(sockfd_ >= 0) {
        ::shutdown(sockfd_, SHUT_RDWR);
        ::close(sockfd_);
        sockfd_ = -1;
    }

    sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if(sockfd_ < 0) {
        return RpcError::INVALID_SOCKET;
    }

    // 设置为非阻塞模式
    RpcError err = set_nonblocking();
    if(err != RpcError::OK) {
        close(sockfd_);
        sockfd_ = -1;
        return err;
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port_);
    if(inet_pton(AF_INET, server_.c_str(), &server_addr.sin_addr) <= 0) {
        close(sockfd_);
        sockfd_ = -1;
        return RpcError::INVALID_ADDRESS;
    }

    // 非阻塞连接
    int ret = ::connect(sockfd_, (struct sockaddr *)&server_addr, sizeof(server_addr));
    if(ret < 0) {
        if(errno == EINPROGRESS) {
            // 等待连接完成
            err = wait_for_writable(CONNECT_TIMEOUT_MS);
            if(err != RpcError::OK) {
                close(sockfd_);
                sockfd_ = -1;
                return err;
            }

            // 检查连接是否成功
            int error = 0;
            socklen_t len = sizeof(error);
            if(getsockopt(sockfd_, SOL_SOCKET, SO_ERROR, &error, &len) < 0 || error != 0) {
                close(sockfd_);
                sockfd_ = -1;
                return RpcError::CONNECTION_CLOSED;
            }
        } else {
            close(sockfd_);
            sockfd_ = -1;
            return RpcError::CONNECTION_CLOSED;
        }
    }

    // 发送握手请求
    HandshakeRequest handshake_req;
    uuid_copy(handshake_req.id, client_id_);
    handshake_req.is_async = is_async_;
    handshake_req.version_key = version_key_;

    err = wait_for_writable(WRITE_TIMEOUT_MS);
    if(err != RpcError::OK) {
        close(sockfd_);
        sockfd_ = -1;
        return err;
    }

    if(::write(sockfd_, &handshake_req, sizeof(handshake_req)) != sizeof(handshake_req)) {
        close(sockfd_);
        sockfd_ = -1;
        return RpcError::WRITE_ERROR;
    }

    // 读取握手响应
    err = wait_for_readable(READ_TIMEOUT_MS);
    if(err != RpcError::OK) {
        close(sockfd_);
        sockfd_ = -1;
        return err;
    }

    HandshakeResponse handshake_rsp;
    if(::read(sockfd_, &handshake_rsp, sizeof(handshake_rsp)) != sizeof(handshake_rsp)) {
        close(sockfd_);
        sockfd_ = -1;
        return RpcError::READ_ERROR;
    }

    if(handshake_rsp.status != 0) {
        close(sockfd_);
        sockfd_ = -1;
        return RpcError::HANDSHAKE_FAILED;
    }

    return RpcError::OK;
}

RpcError RpcConn::disconnect() {
    running_ = false;
    if(sockfd_ >= 0) {
        shutdown(sockfd_, SHUT_RDWR);
        close(sockfd_);
        sockfd_ = -1;
    }
    return RpcError::OK;
}

void RpcConn::prepare_request(uint32_t func_id) {

    func_id_ = func_id;
    iov_send_.clear();
    iov_send2_.clear();
    iov_read_.clear();
    iov_read2_.clear();

    // 添加函数ID到发送缓冲区
    iov_send_.push_back({&func_id_, sizeof(func_id_)});
}

void RpcConn::write(const void *data, size_t len, bool with_len) {
    if(with_len) {
        iov_send2_.push_back({const_cast<void *>(data), len});
    } else {
        iov_send_.push_back({const_cast<void *>(data), len});
    }
}

void RpcConn::read(void *buffer, size_t len, bool with_len) {
    if(with_len) {
        iov_read2_.push_back({buffer, len});
    } else {
        iov_read_.push_back({buffer, len});
    }
}

RpcError RpcConn::submit_request() {
    // 发送数据
    RpcError err = write_all(false);
    if(err != RpcError::OK) {
        return err;
    }
    err = write_all(true);
    if(err != RpcError::OK) {
        return err;
    }

    // 读取响应
    err = read_all(false);
    if(err != RpcError::OK) {
        return err;
    }
    err = read_all(true);
    if(err != RpcError::OK) {
        return err;
    }

    return RpcError::OK;
}

RpcError RpcConn::prepare_response() {
    // 读取请求
    RpcError err = read_all(false);
    if(err != RpcError::OK) {
        return err;
    }
    err = read_all(true);
    if(err != RpcError::OK) {
        return err;
    }
    return RpcError::OK;
}

RpcError RpcConn::submit_response() {
    // 发送响应
    RpcError err = write_all(false);
    if(err != RpcError::OK) {
        return err;
    }
    err = write_all(true);
    if(err != RpcError::OK) {
        return err;
    }

    return RpcError::OK;
}

RpcError RpcConn::write_full_iovec(std::vector<iovec> &iov) {
    if(iov.empty())
        return RpcError::OK;

    size_t remaining = iov.size();
    size_t current = 0;

    while(remaining > 0 && running_) {
        RpcError err = wait_for_writable(WRITE_TIMEOUT_MS);
        if(err != RpcError::OK) {
            if(err == RpcError::WRITE_TIMEOUT) {
                continue;
            }
            return err;
        }

        ssize_t bytes_wrote = ::writev(sockfd_, &iov[current], remaining);
        if(bytes_wrote < 0) {
            if(errno == EINTR)
                continue;
            return RpcError::WRITE_ERROR;
        }
#ifdef DUMP
        size_t to_dump = bytes_wrote;
        for(size_t i = 0; i < remaining; i++) {
            ssize_t dump_len = to_dump > iov[current + i].iov_len ? iov[current + i].iov_len : to_dump;
            hexdump("<==", iov[current + i].iov_base, dump_len);
            to_dump -= dump_len;
        }
#endif
        while(bytes_wrote > 0 && remaining > 0 && running_) {
            if(bytes_wrote >= static_cast<ssize_t>(iov[current].iov_len)) {
                bytes_wrote -= iov[current].iov_len;
                current++;
                remaining--;
            } else {
                iov[current].iov_base = static_cast<char *>(iov[current].iov_base) + bytes_wrote;
                iov[current].iov_len -= bytes_wrote;
                bytes_wrote = 0;
            }
        }
    }
    return remaining == 0 ? RpcError::OK : RpcError::QUIT;
}

RpcError RpcConn::read_full_iovec(std::vector<iovec> &iov) {
    if(iov.empty())
        return RpcError::OK;

    ssize_t total_bytes = 0;
    size_t remaining = iov.size();
    size_t current = 0;

    while(remaining > 0 && running_) {
        RpcError err = wait_for_readable(READ_TIMEOUT_MS);
        if(err != RpcError::OK) {
            if(err == RpcError::READ_TIMEOUT) {
                continue;
            }
            return err;
        }

        ssize_t bytes_read = readv(sockfd_, &iov[current], remaining);
        if(bytes_read < 0) {
            if(errno == EINTR)
                continue;
            return RpcError::READ_ERROR;
        }

        if(bytes_read == 0) {
            return RpcError::CONNECTION_CLOSED;
        }

#ifdef DUMP
        size_t to_dump = bytes_read;
        for(size_t i = 0; i < remaining; i++) {
            ssize_t dump_len = to_dump > iov[current + i].iov_len ? iov[current + i].iov_len : to_dump;
            hexdump("==>", iov[current + i].iov_base, dump_len);
            to_dump -= dump_len;
        }
#endif

        total_bytes += bytes_read;

        while(bytes_read > 0 && remaining > 0 && running_) {
            if(bytes_read >= static_cast<ssize_t>(iov[current].iov_len)) {
                bytes_read -= iov[current].iov_len;
                current++;
                remaining--;
            } else {
                iov[current].iov_base = static_cast<char *>(iov[current].iov_base) + bytes_read;
                iov[current].iov_len -= bytes_read;
                bytes_read = 0;
            }
        }
    }
    return remaining == 0 ? RpcError::OK : RpcError::QUIT;
}

RpcError RpcConn::write_all(bool with_len) {
    if(with_len) {
        std::vector<iovec> iovs;
        for(auto &iov : iov_send2_) {
            iovs.push_back({&iov.iov_len, sizeof(iov.iov_len)});
            iovs.push_back(iov);
        }
        RpcError err = write_full_iovec(iovs);
        iov_send2_.clear();
        return err;
    } else {
        RpcError err = write_full_iovec(iov_send_);
        iov_send_.clear();
        return err;
    }
}

RpcError RpcConn::read_all(bool with_len) {
    if(with_len) {
        for(auto &iov : iov_read2_) {
            // 读取长度字段
            size_t length = 0;
            std::vector<iovec> iovs = {{&length, sizeof(length)}};
            RpcError err = read_full_iovec(iovs);
            if(err != RpcError::OK) {
                return err;
            }

            if(length == 0) {
                continue;
            }

            void *tmp_buffer = nullptr;
            void **buffer_ptr;
            if(iov.iov_len == 0) {
                tmp_buffer = RpcBuffers::getInstance().malloc_rpc_buffer(client_id_str_, length);
                if(tmp_buffer == nullptr) {
                    throw RpcMemoryException("Failed to allocate memory", __LINE__);
                }
                buffer_ptr = static_cast<void **>(iov.iov_base);
                *buffer_ptr = tmp_buffer;

                iov.iov_base = tmp_buffer;
            } else if(iov.iov_len < length) {
                return RpcError::READ_ERROR;
            }
            iov.iov_len = length;

            // Read data
            iovs = {iov};
            err = read_full_iovec(iovs);
            if(err != RpcError::OK) {
                if(tmp_buffer != nullptr) {
                    RpcBuffers::getInstance().free_rpc_buffer(client_id_str_, tmp_buffer);
                    *buffer_ptr = nullptr;
                }
                return err;
            }
        }
        iov_read2_.clear();
        return RpcError::OK;
    }
    RpcError err = read_full_iovec(iov_read_);
    iov_read_.clear();
    return err;
}

RpcError RpcConn::read_one_now(void *buffer, size_t size, bool with_len) {
    size_t length = size;
    if(with_len) {
        // 读取长度字段
        std::vector<iovec> iovs = {{&length, sizeof(length)}};
        RpcError err = read_full_iovec(iovs);

        if(err != RpcError::OK) {
            return err;
        }

        // 检查缓冲区大小
        if(size > 0 && length > size) {
            throw RpcException("Not enough space to read data", __LINE__);
        }
    }

    if(length == 0) {
        return RpcError::OK;
    }

    void *tmp_buffer = nullptr;
    if(size == 0) {
        // 动态分配缓冲区
        tmp_buffer = RpcBuffers::getInstance().malloc_rpc_buffer(client_id_str_, length);
        if(tmp_buffer == nullptr) {
            throw RpcMemoryException("Failed to allocate memory", __LINE__);
        }
        *(void **)buffer = tmp_buffer;
    } else {
        tmp_buffer = buffer;
    }

    std::vector<iovec> iovs = {{tmp_buffer, length}};
    RpcError err = read_full_iovec(iovs);
    if(err != RpcError::OK) {
        if(size == 0) {
            RpcBuffers::getInstance().free_rpc_buffer(client_id_str_, tmp_buffer);
            *(void **)buffer = nullptr;
        }
        return err;
    }

    return RpcError::OK;
}

RpcError RpcConn::read_all_now(void **buffer, size_t *size, int count) {
    for(int i = 0; i < count; i++) {
        // 读取长度字段
        size_t length;
        std::vector<iovec> iovs = {{&length, sizeof(length)}};
        RpcError err = read_full_iovec(iovs);
        if(err != RpcError::OK) {
            return err;
        }

        // 分配新的缓冲区
        void *tmp_buffer = buffer[i];
        if(tmp_buffer == nullptr) {
            tmp_buffer = RpcBuffers::getInstance().malloc_rpc_buffer(client_id_str_, length);
            if(tmp_buffer == nullptr) {
                return RpcError::MALLOC_FAILED;
            }
            buffer[i] = tmp_buffer;
        }
        iovs = {{tmp_buffer, length}};
        err = read_full_iovec(iovs);
        if(err != RpcError::OK) {
            return err;
        }

        if(size != nullptr) {
            size[i] = length;
        }
    }

    return RpcError::OK;
}

void RpcConn::reset() {
    iov_send_.clear();
    iov_send2_.clear();
    iov_read_.clear();
    iov_read2_.clear();
}

void RpcConn::hexdump(const char *desc, const void *buf, size_t len) {
    const unsigned char *p = static_cast<const unsigned char *>(buf);
    printf("\033[%dm[%c]\033[0m %s len: %lu\n", is_server ? 35 : 33, is_server ? 'S' : 'C', desc, len);
    int total_lines = len / 16;
    if(len % 16 != 0) {
        total_lines++;
    }
    int printed_lines = 0;
    // Print 16 bytes per line
    for(size_t i = 0; i < len; i += 16) {
        if(printed_lines == 50) {
            printf("... ... ... ...\n");
        } else if(printed_lines < 50 || printed_lines > total_lines - 50) {
            printf("%08lx: ", i);
            for(size_t j = 0; j < 16; j++) {
                if(i + j < len) {
                    printf("%02x ", p[i + j]);
                } else {
                    printf("   ");
                }
            }
            printf(" ");
            for(size_t j = 0; j < 16; j++) {
                if(i + j < len) {
                    printf("%c", isprint(p[i + j]) ? p[i + j] : '.');
                }
            }
            printf("\n");
        }
        printed_lines++;
    }
}

void *RpcConn::get_host_buffer(size_t size) { return RpcBuffers::getInstance().malloc_rpc_buffer(client_id_str_, size); }

void RpcConn::free_host_buffer(void *ptr) { RpcBuffers::getInstance().free_rpc_buffer(client_id_str_, ptr); }

void *RpcConn::get_iov_buffer(size_t size) {
    std::lock_guard<std::mutex> lock(iov_buffers_mutex_);
    void *iov_buffer = malloc(size);
    if(iov_buffer == nullptr) {
        return nullptr;
    }
    memset(iov_buffer, 0, size);
    iov_buffers_.insert(iov_buffer);
    return iov_buffer;
}

void RpcConn::free_iov_buffer(void *ptr) {
    std::lock_guard<std::mutex> lock(iov_buffers_mutex_);
    if(ptr == nullptr) {
        return;
    }
    auto it = iov_buffers_.find(ptr);
    if(it == iov_buffers_.end()) {
        return;
    }
    iov_buffers_.erase(it);
    free(ptr);
}

void RpcConn::free_all_iov_buffers() {
    std::lock_guard<std::mutex> lock(iov_buffers_mutex_);
    for(auto ptr : iov_buffers_) {
        free(ptr);
    }
    iov_buffers_.clear();
}

} // namespace rpc
