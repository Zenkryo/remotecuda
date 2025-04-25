#include "rpc_core.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>

using namespace rpc;

// 示例处理函数
int echo_handler(RpcConn *conn) {
    char buffer[1024];
    char name[1024];
    size_t len;

    // 读取数据长度
    conn->read(&len, sizeof(len));
    conn->prepare_response();
    if(len > sizeof(buffer)) {
        return -1;
    }

    // 读取数据
    RpcError err = conn->read_one_now(buffer, len);
    if(err != RpcError::OK) {
        return -1;
    }
    err = conn->read_one_now(name, sizeof(name), true);
    if(err != RpcError::OK) {
        return -1;
    }

    // 发送响应
    conn->write(&len, sizeof(len));
    conn->write(buffer, len);
    conn->write(name, strlen(name) + 1, true);
    err = conn->submit_response();
    if(err != RpcError::OK) {
        return -1;
    }

    return 0;
}

// 数值计算服务示例
int calculate_handler(RpcConn *conn) {
    int operation;
    int a, b, result;
    conn->read(&operation, sizeof(operation));
    conn->read(&a, sizeof(a));
    conn->read(&b, sizeof(b));

    conn->prepare_response();
    switch(operation) {
    case 1: // 加法
        result = a + b;
        break;
    case 2: // 减法
        result = a - b;
        break;
    case 3: // 乘法
        result = a * b;
        break;
    case 4: // 除法
        if(b == 0) {
            result = 0;
            conn->write(&result, sizeof(result));
            RpcError err = conn->submit_response();
            if(err != RpcError::OK) {
                return -1;
            }
            return -1;
        }
        result = a / b;
        break;
    default:
        result = 0;
        conn->write(&result, sizeof(result));
        RpcError err = conn->submit_response();
        if(err != RpcError::OK) {
            return -1;
        }
        return -1;
    }

    conn->write(&result, sizeof(result));
    RpcError err = conn->submit_response();
    if(err != RpcError::OK) {
        return -1;
    }
    return 0;
}

// 服务器端示例
void run_server() {
    try {
        RpcServer server(12345, VERSION_KEY);

        // 注册处理函数
        server.register_handler(1, echo_handler);
        server.register_handler(2, calculate_handler);

        std::cout << "Server starting..." << std::endl;
        server.start();
        printf("server start\n");
    } catch(const RpcException &e) {
        std::cerr << "Server error: " << e.what() << std::endl;
    }
}

void case1(RpcClient &client) {
    // 1. 测试echo服务
    RpcConn *conn = nullptr;
    while(!conn) {
        conn = client.acquire_connection();
        if(!conn) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    conn->prepare_request(1);
    const char *message = "Hello, RPC!";
    const char *name = "tester";

    size_t len = strlen(message) + 1;
    conn->write(&len, sizeof(len));
    conn->write(message, len);
    conn->write(name, strlen(name) + 1, true);

    char message_echo[1024];
    conn->read(message_echo, sizeof(message_echo), true);
    char name_echo[1024];
    conn->read(name_echo, sizeof(name_echo), true);

    // 提交请求
    RpcError err = conn->submit_request();
    if(err != RpcError::OK) {
        std::cerr << "Echo request failed: " << static_cast<int>(err) << std::endl;
        client.release_connection(conn);
        return;
    }

    std::cout << "Echo response: " << std::string(message_echo) << " name" << std::string(name_echo) << std::endl;
    client.release_connection(conn);
}

void case2(RpcClient &client) {
    // 2. 测试计算服务
    RpcConn *conn = nullptr;
    while(!conn) {
        conn = client.acquire_connection();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    conn->prepare_request(2);
    int operation = 1; // 加法
    int a = 10, b = 20;
    conn->write(&operation, sizeof(operation));
    conn->write(&a, sizeof(a));
    conn->write(&b, sizeof(b));

    int result;
    conn->read(&result, sizeof(result));

    // 提交请求
    RpcError err = conn->submit_request();
    if(err != RpcError::OK) {
        std::cerr << "Calculation request failed: " << static_cast<int>(err) << std::endl;
        client.release_connection(conn);
        return;
    }

    std::cout << "Calculation result: " << result << std::endl;
    client.release_connection(conn);
}

// 客户端示例
void run_client() {
    try {
        RpcClient client(VERSION_KEY);

        // 连接到服务器
        RpcError err = client.connect("127.0.0.1", 12345, 1);
        if(err != RpcError::OK) {
            std::cerr << "Failed to connect: " << static_cast<int>(err) << std::endl;
            return;
        }

        case1(client);

        // // 创建20个线程
        std::vector<std::thread> threads;
        for(int i = 0; i < 20; i++) {
            threads.emplace_back([&client]() {
                case1(client);
                case2(client);
            });
        }

        // 等待所有线程完成
        for(auto &thread : threads) {
            thread.join();
        }

        // client.disconnect();
    } catch(const RpcException &e) {
        std::cerr << "Client error: " << e.what() << std::endl;
    }
    printf("run_client quit\n");
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
