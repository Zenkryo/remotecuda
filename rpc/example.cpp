#include "rpc_core.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>

using namespace rpc;

// 示例处理函数
int echo_handler(RpcClient *client) {
    char buffer[1024];
    size_t len;

    // 读取数据长度
    client->read(&len, sizeof(len));
    client->prepare_response();
    if(len > sizeof(buffer)) {
        return -1;
    }

    // 读取数据
    client->read_one_now(buffer, len);
    // 发送响应
    client->write(&len, sizeof(len));
    client->write(buffer, len);
    client->submit_response();

    return 0;
}

// 服务器端示例
void run_server() {
    try {
        RpcServer server(12345, 0);

        // 注册处理函数
        server.register_handler(1, echo_handler);

        std::cout << "Server starting..." << std::endl;
        server.start();
    } catch(const RpcException &e) {
        std::cerr << "Server error: " << e.what() << std::endl;
    }
}

// 客户端示例
void run_client() {
    try {
        RpcClient client;

        // 连接到服务器
        client.connect("127.0.0.1", 12345);

        // 准备请求
        client.prepare_request(1);

        // 发送数据
        const char *message = "Hello, RPC!";
        size_t len = strlen(message);
        client.write(&len, sizeof(len) + 1);
        client.write(message, len);

        // 读取响应
        size_t response_len;
        client.read(&response_len, sizeof(response_len));

        char response[1024];
        // client.read(response, response_len);

        // 提交请求
        if(client.submit_request() != 0) {
            throw RpcException("Request failed");
        }
        std::cout << "response_len: " << response_len << std::endl;
        client.read_one_now(response, response_len);
        std::cout << "Received: " << std::string(response, response_len) << std::endl;

    } catch(const RpcException &e) {
        std::cerr << "Client error: " << e.what() << std::endl;
    }
}

int main() {
    // 启动服务器线程
    std::thread server_thread(run_server);

    // 等待服务器启动
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 运行客户端
    run_client();

    // 等待服务器线程结束
    server_thread.join();

    return 0;
}
