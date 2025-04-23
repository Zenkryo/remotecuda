#include "rpc_core.h"
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace rpc {

RpcClient::RpcClient() : sockfd_(-1), func_id_(0), in_use_(false) { uuid_generate(client_id_); }

RpcClient::~RpcClient() {
    disconnect();
    cleanup_tmp_buffers();
}

void RpcClient::connect(const std::string &server, uint16_t port, bool is_async) {
    std::lock_guard<std::mutex> lock(mutex_);

    if(sockfd_ >= 0) {
        throw RpcException("Client already connected");
    }

    sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if(sockfd_ < 0) {
        throw RpcException("Failed to create socket: " + std::string(strerror(errno)));
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    if(inet_pton(AF_INET, server.c_str(), &server_addr.sin_addr) <= 0) {
        close(sockfd_);
        sockfd_ = -1;
        throw RpcException("Invalid address: " + server);
    }

    if(::connect(sockfd_, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        close(sockfd_);
        sockfd_ = -1;
        throw RpcException("Connection failed: " + std::string(strerror(errno)));
    }

    // 发送握手请求
    HandshakeRequest handshake_req;
    uuid_copy(handshake_req.id, client_id_);
    handshake_req.is_async = is_async;
    handshake_req.version_key = 0; // TODO: 从配置获取

    if(::write(sockfd_, &handshake_req, sizeof(handshake_req)) != sizeof(handshake_req)) {
        close(sockfd_);
        sockfd_ = -1;
        throw RpcException("Failed to send handshake request");
    }

    // 读取握手响应
    HandshakeResponse handshake_rsp;
    if(::read(sockfd_, &handshake_rsp, sizeof(handshake_rsp)) != sizeof(handshake_rsp)) {
        close(sockfd_);
        sockfd_ = -1;
        throw RpcException("Failed to read handshake response");
    }

    if(handshake_rsp.status != 0) {
        close(sockfd_);
        sockfd_ = -1;
        throw RpcException("Handshake failed");
    }
}

void RpcClient::disconnect() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(sockfd_ >= 0) {
        shutdown(sockfd_, SHUT_RDWR);
        close(sockfd_);
        sockfd_ = -1;
        cleanup_tmp_buffers();
    }
}

void RpcClient::prepare_request(uint32_t func_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    func_id_ = func_id;
    iov_send_.clear();
    iov_send2_.clear();
    iov_read_.clear();
    iov_read2_.clear();

    // 添加函数ID到发送缓冲区
    iov_send_.push_back({&func_id_, sizeof(func_id_)});
}

void RpcClient::write(const void *data, size_t len, bool with_len) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(with_len) {
        iov_send2_.push_back({const_cast<void *>(data), len});
    } else {
        iov_send_.push_back({const_cast<void *>(data), len});
    }
}

void RpcClient::read(void *buffer, size_t len, bool with_len) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(with_len) {
        iov_read2_.push_back({buffer, len});
    } else {
        iov_read_.push_back({buffer, len});
    }
}

int RpcClient::submit_request() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(sockfd_ < 0) {
        throw RpcException("Not connected");
    }

    // 发送数据
    if(write_full_iovec(iov_send_) < 0 || write_full_iovec(iov_send2_) < 0) {
        throw RpcException("Failed to send request: " + std::string(strerror(errno)));
    }

    // 读取响应
    if(read_full_iovec(iov_read_) < 0 || read_full_iovec(iov_read2_) < 0) {
        throw RpcException("Failed to read response: " + std::string(strerror(errno)));
    }

    return 0;
}

int RpcClient::prepare_response() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(sockfd_ < 0) {
        throw RpcException("Not connected");
    }

    // 读取请求
    if(read_full_iovec(iov_read_) < 0 || read_full_iovec(iov_read2_) < 0) {
        throw RpcException("Failed to read request: " + std::string(strerror(errno)));
    }

    return 0;
}

int RpcClient::submit_response() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(sockfd_ < 0) {
        throw RpcException("Not connected");
    }

    // 发送响应
    if(write_full_iovec(iov_send_) < 0 || write_full_iovec(iov_send2_) < 0) {
        throw RpcException("Failed to send response: " + std::string(strerror(errno)));
    }

    return 0;
}

ssize_t RpcClient::write_full_iovec(std::vector<iovec> &iov) {
    if(iov.empty())
        return 0;

    ssize_t total_bytes = 0;
    size_t remaining = iov.size();
    size_t current = 0;

    while(remaining > 0) {
        ssize_t bytes_written = writev(sockfd_, &iov[current], remaining);
        if(bytes_written < 0) {
            if(errno == EINTR)
                continue;
            return -1;
        }

        total_bytes += bytes_written;

        // 调试输出
        std::cout << "Write " << bytes_written << " bytes:" << std::endl;
        for(size_t i = 0; i < remaining && i < 3; i++) {
            std::cout << "  iov[" << i << "]: ";
            for(size_t j = 0; j < std::min(iov[current + i].iov_len, size_t(32)); j++) {
                printf("%02x ", static_cast<unsigned char *>(iov[current + i].iov_base)[j]);
            }
            std::cout << std::endl;
        }

        // 调整iovec数组
        while(bytes_written > 0 && remaining > 0) {
            if(bytes_written >= static_cast<ssize_t>(iov[current].iov_len)) {
                bytes_written -= iov[current].iov_len;
                current++;
                remaining--;
            } else {
                iov[current].iov_base = static_cast<char *>(iov[current].iov_base) + bytes_written;
                iov[current].iov_len -= bytes_written;
                bytes_written = 0;
            }
        }
    }

    return total_bytes;
}

ssize_t RpcClient::read_full_iovec(std::vector<iovec> &iov) {
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
            return -1; // 连接关闭
        }

        total_bytes += bytes_read;

        // 调试输出
        std::cout << "Read " << bytes_read << " bytes:" << std::endl;
        for(size_t i = 0; i < remaining && i < 3; i++) {
            std::cout << "  iov[" << i << "]: ";
            for(size_t j = 0; j < std::min(iov[current + i].iov_len, size_t(32)); j++) {
                printf("%02x ", static_cast<unsigned char *>(iov[current + i].iov_base)[j]);
            }
            std::cout << std::endl;
        }

        // 调整iovec数组
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

ssize_t RpcClient::read_one_now(void *buffer, size_t size, bool with_len) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(sockfd_ < 0) {
        throw RpcException("Not connected");
    }

    ssize_t total_read = 0;
    size_t length = size;

    if(with_len) {
        // 读取长度字段
        if(::read(sockfd_, &length, sizeof(length)) != sizeof(length)) {
            return -1;
        }
        total_read += sizeof(length);

        // 检查缓冲区大小
        if(size > 0 && length > size) {
            errno = ENOBUFS;
            return -1;
        }
    }

    if(length == 0) {
        return total_read;
    }

    // 读取数据
    if(size == 0) {
        // 动态分配缓冲区
        void *tmp_buffer = malloc(length);
        if(tmp_buffer == nullptr) {
            errno = ENOBUFS;
            return -1;
        }
        *(void **)buffer = tmp_buffer;
        tmp_buffers_.insert(tmp_buffer);
    }

    ssize_t bytes_read = ::read(sockfd_, buffer, length);
    if(bytes_read < 0) {
        if(size == 0) {
            free(*(void **)buffer);
            tmp_buffers_.erase(*(void **)buffer);
        }
        return -1;
    }

    total_read += bytes_read;
    return total_read;
}

ssize_t RpcClient::read_all_now(void **buffer, size_t *size, int count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(sockfd_ < 0) {
        throw RpcException("Not connected");
    }

    ssize_t total_read = 0;
    std::set<void *> new_buffers; // 用于跟踪新分配的缓冲区

    for(int i = 0; i < count; i++) {
        // 读取长度字段
        size_t length;
        if(::read(sockfd_, &length, sizeof(length)) != sizeof(length)) {
            // 发生错误，清理已分配的缓冲区
            for(void *ptr : new_buffers) {
                free(ptr);
            }
            return -1;
        }
        total_read += sizeof(length);

        // 分配新的缓冲区
        void *new_buffer = malloc(length);
        if(new_buffer == nullptr) {
            // 内存分配失败，清理已分配的缓冲区
            for(void *ptr : new_buffers) {
                free(ptr);
            }
            errno = ENOBUFS;
            return -1;
        }
        new_buffers.insert(new_buffer);

        // 读取数据
        ssize_t bytes_read = ::read(sockfd_, new_buffer, length);
        if(bytes_read < 0) {
            // 读取失败，清理已分配的缓冲区
            for(void *ptr : new_buffers) {
                free(ptr);
            }
            return -1;
        }
        total_read += bytes_read;

        // 保存缓冲区指针和大小
        buffer[i] = new_buffer;
        if(size != nullptr) {
            size[i] = length;
        }
    }

    // 将新分配的缓冲区添加到tmp_buffers_集合中
    tmp_buffers_.insert(new_buffers.begin(), new_buffers.end());

    return total_read;
}

void RpcClient::cleanup_tmp_buffers() {
    for(void *ptr : tmp_buffers_) {
        free(ptr);
    }
    tmp_buffers_.clear();
}

} // namespace rpc
