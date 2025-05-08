#include "rpc_core.h"
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <fcntl.h>
#include <sys/select.h>
#include <errno.h>

namespace rpc {

void *RpcBuffers::malloc_rpc_buffer(std::string client_id, size_t size) {
    std::lock_guard<std::mutex> lock(buffers_mutex_);
    void *raw_ptr = malloc(size);
    if(raw_ptr == nullptr) {
        return nullptr;
    }
    memset(raw_ptr, 0, size);
    rpc_buffers_[client_id].insert(raw_ptr);
    return raw_ptr;
}

void RpcBuffers::free_rpc_buffer(std::string client_id, void *ptr) {
    std::lock_guard<std::mutex> lock(buffers_mutex_);
    auto it = rpc_buffers_.find(client_id);
    if(it == rpc_buffers_.end()) {
        return;
    }
    auto &buffers = it->second;
    for(auto it2 = buffers.begin(); it2 != buffers.end(); ++it2) {
        if(*it2 == ptr) {
            free(*it2);
            buffers.erase(it2);
            return;
        }
    }
}

void RpcBuffers::increment_connection_count(const std::string &client_id) {
    std::lock_guard<std::mutex> lock(buffers_mutex_);
    connection_counts_[client_id]++;
}

void RpcBuffers::decrement_connection_count(const std::string &client_id) {
    std::lock_guard<std::mutex> lock(buffers_mutex_);
    auto it = connection_counts_.find(client_id);
    if(it == connection_counts_.end()) {
        return;
    }

    if(--it->second <= 0) {
        // 当连接计数为0时，清理该client的所有缓冲区
        auto buffers_it = rpc_buffers_.find(client_id);
        if(buffers_it != rpc_buffers_.end()) {
            for(auto ptr : buffers_it->second) {
                free(ptr);
            }
            rpc_buffers_.erase(buffers_it);
        }
        connection_counts_.erase(it);
    }
}

} // namespace rpc
