cmake_minimum_required(VERSION 3.10)
project(rpc_framework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找必要的包
find_package(Threads REQUIRED)

# 添加源文件
set(SOURCES
    rpc_client.cpp
    rpc_server.cpp
    example.cpp
)

# 添加头文件
set(HEADERS
    rpc_core.h
)

# 创建可执行文件
add_executable(rpc_example ${SOURCES} ${HEADERS})

# 链接必要的库
target_link_libraries(rpc_example
    PRIVATE
    Threads::Threads
    uuid
)

# 添加编译选项
target_compile_options(rpc_example
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
