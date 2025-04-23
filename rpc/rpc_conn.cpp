#include "rpc_core.h"
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace rpc {

RpcConn::RpcConn(uint16_t version_key, uuid_t client_id, bool is_server) : sockfd_(-1), func_id_(0), client_id_(), version_key_(version_key), is_server(is_server) { uuid_copy(client_id_, client_id); }

RpcConn::~RpcConn() {
    disconnect();
    cleanup_tmp_buffers();
}

void RpcConn::connect(const std::string &server, uint16_t port, bool is_async) {
    std::lock_guard<std::mutex> lock(mutex_);

    if(sockfd_ >= 0) {
        throw RpcException("Client already connected", __LINE__);
    }

    sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if(sockfd_ < 0) {
        throw RpcException("Failed to create socket: " + std::string(strerror(errno)), __LINE__);
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    if(inet_pton(AF_INET, server.c_str(), &server_addr.sin_addr) <= 0) {
        close(sockfd_);
        sockfd_ = -1;
        throw RpcException("Invalid address: " + server, __LINE__);
    }

    if(::connect(sockfd_, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        close(sockfd_);
        sockfd_ = -1;
        throw RpcException("Connection failed: " + std::string(strerror(errno)), __LINE__);
    }

    // 发送握手请求
    HandshakeRequest handshake_req;
    uuid_copy(handshake_req.id, client_id_);
    handshake_req.is_async = is_async;
    handshake_req.version_key = version_key_;

    if(::write(sockfd_, &handshake_req, sizeof(handshake_req)) != sizeof(handshake_req)) {
        close(sockfd_);
        sockfd_ = -1;
        throw RpcException("Failed to send handshake request", __LINE__);
    }

    // 读取握手响应
    HandshakeResponse handshake_rsp;
    if(::read(sockfd_, &handshake_rsp, sizeof(handshake_rsp)) != sizeof(handshake_rsp)) {
        close(sockfd_);
        sockfd_ = -1;
        throw RpcException("Failed to read handshake response", __LINE__);
    }

    if(handshake_rsp.status != 0) {
        close(sockfd_);
        sockfd_ = -1;
        throw RpcException("Handshake failed", __LINE__);
    }
}

void RpcConn::disconnect() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(sockfd_ >= 0) {
        shutdown(sockfd_, SHUT_RDWR);
        close(sockfd_);
        sockfd_ = -1;
        cleanup_tmp_buffers();
    }
}

void RpcConn::prepare_request(uint32_t func_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    func_id_ = func_id;
    iov_send_.clear();
    iov_send2_.clear();
    iov_read_.clear();
    iov_read2_.clear();

    // 添加函数ID到发送缓冲区
    iov_send_.push_back({&func_id_, sizeof(func_id_)});
}

void RpcConn::write(const void *data, size_t len, bool with_len) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(with_len) {
        iov_send2_.push_back({const_cast<void *>(data), len});
    } else {
        iov_send_.push_back({const_cast<void *>(data), len});
    }
}

void RpcConn::read(void *buffer, size_t len, bool with_len) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(with_len) {
        iov_read2_.push_back({buffer, len});
    } else {
        iov_read_.push_back({buffer, len});
    }
}

void RpcConn::submit_request() {
    std::lock_guard<std::mutex> lock(mutex_);
    // 发送数据
    if(write_all(false) < 0 || write_all(true) < 0) {
        throw RpcConnException("Failed to send request: " + std::string(strerror(errno)), __LINE__);
    }

    // 读取响应
    if(read_all(false) < 0 || read_all(true) < 0) {
        throw RpcConnException("Failed to read response: " + std::string(strerror(errno)), __LINE__);
    }
}

void RpcConn::prepare_response() {
    std::lock_guard<std::mutex> lock(mutex_);
    // 读取请求
    if(read_all(false) < 0 || read_all(true) < 0) {
        throw RpcConnException("Failed to read request: " + std::string(strerror(errno)), __LINE__);
    }
}

void RpcConn::submit_response() {
    std::lock_guard<std::mutex> lock(mutex_);
    // 发送响应
    if(write_all(false) < 0 || write_all(true) < 0) {
        throw RpcConnException("Failed to send response: " + std::string(strerror(errno)), __LINE__);
    }
}

ssize_t RpcConn::write_full_iovec(std::vector<iovec> &iov) {
    if(iov.empty())
        return 0;

    ssize_t total_bytes = 0;
    size_t remaining = iov.size();
    size_t current = 0;

    while(remaining > 0) {
        ssize_t bytes_wrote = ::writev(sockfd_, &iov[current], remaining);
        if(bytes_wrote < 0) {
            if(errno == EINTR)
                continue;
            return -1;
        }

        total_bytes += bytes_wrote;

#ifdef DUMP
        size_t bytes_remaining = bytes_wrote;
        for(size_t i = current; i < current + iov.size() && bytes_remaining > 0; i++) {
            size_t len = std::min(iov[i].iov_len, static_cast<size_t>(bytes_remaining));
            RpcConn::hexdump("<== ", iov[i].iov_base, len);
            bytes_remaining -= len;
        }
#endif
        while(bytes_wrote > 0 && remaining > 0) {
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
    return total_bytes;
}

ssize_t RpcConn::read_full_iovec(std::vector<iovec> &iov) {
    if(iov.empty())
        return 0;

    ssize_t total_bytes = 0;
    size_t remaining = iov.size();
    size_t current = 0;

    while(remaining > 0) {
        ssize_t bytes_read = readv(sockfd_, &iov[current], remaining);
        if(bytes_read < 0) {
            if(errno == EINTR)
                continue;
            return -1;
        }

        if(bytes_read == 0) {
            return -1; // Connection closed
        }

        total_bytes += bytes_read;

#ifdef DUMP
        size_t bytes_remaining = bytes_read;
        for(size_t i = current; i < current + iov.size() && bytes_remaining > 0; i++) {
            size_t len = std::min(iov[i].iov_len, static_cast<size_t>(bytes_remaining));
            RpcConn::hexdump("==> ", iov[i].iov_base, len);
            bytes_remaining -= len;
        }
#endif

        while(bytes_read > 0 && remaining > 0) {
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
    return total_bytes;
}

ssize_t RpcConn::write_all(bool with_len = false) {
    ssize_t bytes_wrote = 0;
    if(with_len) {
        std::vector<iovec> iovs;
        for(auto &iov : iov_send2_) {
            iovs.push_back({&iov.iov_len, sizeof(iov.iov_len)});
            iovs.push_back(iov);
        }
        bytes_wrote = write_full_iovec(iovs);
        iov_send2_.clear();
    } else {
        bytes_wrote = write_full_iovec(iov_send_);
        iov_send_.clear();
    }
    return bytes_wrote;
}

ssize_t RpcConn::read_all(bool with_len = false) {
    ssize_t total_read = 0;
    ssize_t bytes_read = 0;
    if(with_len) {
        for(auto &iov : iov_read2_) {
            // 读取长度字段
            size_t length = 0;
            std::vector<iovec> iovs = {{&length, sizeof(length)}};
            bytes_read = read_full_iovec(iovs);
            if(bytes_read < 0) {
                return -1;
            }
            total_read += bytes_read;
            if(length == 0) {
                continue;
            }
            void *tmp_buffer = nullptr;
            if(iov.iov_len == 0) {
                tmp_buffer = malloc(length);
                if(tmp_buffer == nullptr) {
                    throw RpcException("Failed to allocate memory: " + std::string(strerror(errno)), __LINE__);
                }
                void **buffer_ptr = static_cast<void **>(iov.iov_base);
                *buffer_ptr = tmp_buffer;
            } else if(iov.iov_len < length) {
                throw RpcException("Buffer too small: " + std::string(strerror(ENOBUFS)), __LINE__);
            }
            iov.iov_len = length;

            // Read data
            iovs = {iov};
            bytes_read = read_full_iovec(iovs);
            if(bytes_read < 0) {
                if(tmp_buffer != nullptr) {
                    free(tmp_buffer);
                    void **buffer_ptr = static_cast<void **>(iov.iov_base);
                    *buffer_ptr = nullptr;
                }
                return -1;
            }
            if(tmp_buffer != nullptr) {
                tmp_buffers_.insert(tmp_buffer);
            }
            total_read += bytes_read;
        }
        iov_read2_.clear();
        return total_read;
    }
    total_read = read_full_iovec(iov_read_);
    iov_read_.clear();
    return total_read;
}

void RpcConn::read_one_now(void *buffer, size_t size, bool with_len) {
    std::lock_guard<std::mutex> lock(mutex_);

    ssize_t total_read = 0;
    ssize_t bytes_read;
    size_t length = size;

    if(with_len) {
        // 读取长度字段
        std::vector<iovec> iovs = {{&length, sizeof(length)}};
        bytes_read = read_full_iovec(iovs);
        if(bytes_read < 0) {
            throw RpcConnException("Failed to read length: " + std::string(strerror(errno)), __LINE__);
        }
        total_read += bytes_read;

        // 检查缓冲区大小
        if(size > 0 && length > size) {
            throw RpcConnException("Buffer too small: " + std::string(strerror(ENOBUFS)), __LINE__);
        }
    }

    if(length == 0) {
        return;
    }
    void *tmp_buffer = nullptr;

    // 读取数据
    if(size == 0) {
        // 动态分配缓冲区
        tmp_buffer = malloc(length);
        if(tmp_buffer == nullptr) {
            throw RpcException("Failed to allocate memory: " + std::string(strerror(errno)), __LINE__);
        }
        *(void **)buffer = tmp_buffer;
    }
    std::vector<iovec> iovs = {{buffer, length}};
    bytes_read = read_full_iovec(iovs);
    if(bytes_read < 0) {
        if(tmp_buffer != nullptr) {
            free(*(void **)buffer);
        }
        throw RpcException("Failed to read data: " + std::string(strerror(errno)), __LINE__);
    }
    if(tmp_buffer != nullptr) {
        tmp_buffers_.insert(tmp_buffer);
    }
}

void RpcConn::read_all_now(void **buffer, size_t *size, int count) {
    std::lock_guard<std::mutex> lock(mutex_);

    ssize_t total_read = 0;
    ssize_t bytes_read;
    std::set<void *> new_buffers; // 用于跟踪新分配的缓冲区

    for(int i = 0; i < count; i++) {
        // 读取长度字段
        size_t length;
        std::vector<iovec> iovs = {{&length, sizeof(length)}};
        bytes_read = read_full_iovec(iovs);
        if(bytes_read < 0) {
            for(void *ptr : new_buffers) {
                free(ptr);
            }
            throw RpcConnException("Failed to read length: " + std::string(strerror(errno)), __LINE__);
        }
        total_read += bytes_read;

        // 分配新的缓冲区
        void *tmp_buffer = malloc(length);
        if(tmp_buffer == nullptr) {
            // 内存分配失败，清理已分配的缓冲区
            for(void *ptr : new_buffers) {
                free(ptr);
            }
            throw RpcException("Failed to allocate memory: " + std::string(strerror(errno)), __LINE__);
        }
        new_buffers.insert(tmp_buffer);
        iovs = {{tmp_buffer, length}};
        // 读取数据
        bytes_read = read_full_iovec(iovs);
        if(bytes_read < 0) {
            for(void *ptr : new_buffers) {
                free(ptr);
            }
            throw RpcConnException("Failed to read data: " + std::string(strerror(errno)), __LINE__);
        }
        total_read += bytes_read;

        // 保存缓冲区指针和大小
        buffer[i] = tmp_buffer;
        if(size != nullptr) {
            size[i] = length;
        }
    }

    // 将新分配的缓冲区添加到tmp_buffers_集合中
    tmp_buffers_.insert(new_buffers.begin(), new_buffers.end());
}

void RpcConn::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    iov_send_.clear();
    iov_send2_.clear();
    iov_read_.clear();
    iov_read2_.clear();
}

void RpcConn::cleanup_tmp_buffers() {
    for(void *ptr : tmp_buffers_) {
        free(ptr);
    }
    tmp_buffers_.clear();
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

} // namespace rpc
