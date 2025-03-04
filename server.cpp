#include "gen/hook_api.h"
#include "gen/handle_server.h"
#include <unordered_map>
#include <iostream>

extern std::unordered_map<uint32_t, RequestHandler> handlerMap;

RequestHandler get_handler(const uint32_t funcId) {
    auto it = handlerMap.find(funcId);
    if(it != handlerMap.end()) {
        return it->second;
    }
    return nullptr;
}
