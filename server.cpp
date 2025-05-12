#include <iostream>
#include <thread>
#include <signal.h>
#include <gen/hook_api.h>
#include "rpc/rpc_core.h"

using namespace rpc;
extern std::map<uint32_t, RequestHandler> handlerMap;

void run_server() {
    try {
        RpcServer &server = RpcServer::getInstance();

        // 注册处理函数
        for(auto it = handlerMap.begin(); it != handlerMap.end(); it++) {
            server.register_handler(it->first, it->second);
        }
        std::cout << "Server starting..." << std::endl;
        server.start(VERSION_KEY);
    } catch(const RpcException &e) {
        std::cerr << "Server error: " << e.what() << std::endl;
    }
}

int main() {
    signal(SIGPIPE, SIG_IGN);
    // 启动服务器线程
    std::thread server_thread(run_server);
    // 等待服务器线程结束
    server_thread.join();
}
