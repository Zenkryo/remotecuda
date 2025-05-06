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
            buffers.erase(it2);
            return;
        }
    }
}

} // namespace rpc
