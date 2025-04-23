cmake_minimum_required(VERSION 3.10)
project(rpc_framework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加DEBUG选项: 用法: cmake -DDEBUG=ON
option(DEBUG "Enable debug mode" OFF)
# 添加DUMP选项: 用法: cmake -DDUMP=ON
option(DUMP "Enable dump mode" OFF)

# 设置打印详细编译命令
set(CMAKE_VERBOSE_MAKEFILE ON)

# 查找必要的包
find_package(Threads REQUIRED)

# 添加源文件
set(SOURCES
    rpc_conn.cpp
    rpc_server.cpp
    rpc_client.cpp
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

# 根据DEBUG选项添加编译定义
if(DEBUG)
    target_compile_definitions(rpc_example PRIVATE DEBUG)
    target_compile_options(rpc_example PRIVATE -g -O0)
else()
    target_compile_options(rpc_example PRIVATE -O2)
endif()

# 根据DUMP选项添加编译定义
if(DUMP)
    target_compile_definitions(rpc_example PRIVATE DUMP)
endif()

# 打印编译命令
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Debug mode: ${DEBUG}")
message(STATUS "Dump mode: ${DUMP}")
