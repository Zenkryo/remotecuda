# RemoteCuda

RemoteCuda 是一个用于远程 CUDA 操作的项目，它允许在没有本地 GPU 的客户端上运行 CUDA 应用程序，通过 RPC 调用远程 GPU 服务器上的 CUDA 资源。该项目通过动态库注入的方式，自动接管本地 CUDA API 调用，将其转换为远程调用。

## 工作原理

RemoteCuda 使用动态库注入技术（LD_PRELOAD）来接管本地 CUDA API 调用。当客户端运行 CUDA 应用程序时，hook.so 会被优先加载，拦截所有 CUDA API 调用，并将其转换为 RPC 请求发送到远程 GPU 服务器。服务器端接收请求后执行实际的 CUDA 操作，并将结果返回给客户端。

## 功能特性

- 支持远程 CUDA 操作
- 实现了完整的 CUDA Runtime API 和 CUDA Driver API 接口
- 支持以下主要功能:
  - 内存管理 (cudaMalloc, cudaFree 等)
  - 数据传输 (cudaMemcpy 系列)
  - 流管理 (cudaStream 系列)
  - 事件管理 (cudaEvent 系列)
  - 模块管理 (cuModule 系列)
  - 函数调用 (cuLaunch 系列)
  - 图形执行 (cudaGraph 系列)
  - 库管理 (cudaLibrary 系列)
  - 纹理引用 (cuTexRef 系列)
  - CUBLAS 操作

## 系统要求

- 服务器端：
  - CUDA 工具包
  - GPU 设备
  - C++ 编译器
  - 网络连接

- 客户端：
  - CUDA 工具包（仅用于编译）
  - C++ 编译器
  - 网络连接

## 使用方法

### 1. 生成 Hook 代码

```bash
# 在项目根目录下执行
./hook.py -O gen/
```

### 2. 编译客户端和服务器端程序

```bash
cd gen/
make
```

编译完成后会在 gen 目录下生成：
- `hook.so` - 客户端使用的动态链接库
- `server` - GPU 服务器端运行的可执行程序

### 3. 启动服务器

在 GPU 服务器上运行：
```bash
./server
```

### 4. 运行客户端程序

在客户端机器上，通过 LD_PRELOAD 加载 hook.so 来运行 CUDA 应用程序：

```bash
LD_PRELOAD=./hook.so ./your_cuda_app
```

示例代码：
```cpp
#include "cuda_runtime.h"

int main() {
    // 分配设备内存
    void* d_ptr;
    cudaMalloc(&d_ptr, size);

    // 数据传输
    cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);

    // 清理
    cudaFree(d_ptr);

    return 0;
}
```

## 项目结构

- `hook.py` - 自动生成 hook 代码的脚本
- `gen/` - 生成的代码目录
  - `handle_cuda.cpp` - CUDA Driver API 处理函数
  - `handle_cuda_runtime_api.cpp` - CUDA Runtime API 处理函数
  - `hook_api.h` - API 钩子定义
  - `Makefile` - 编译配置文件
- `rpc/` - RPC 相关代码
- `hidden_api.h` - 隐藏的 CUDA API 定义

## 注意事项

1. 确保服务器和客户端都安装了相同版本的 CUDA
2. 网络连接质量会影响性能
3. 某些 CUDA 操作可能需要特殊权限
4. 客户端程序需要使用与服务器端相同版本的 CUDA 工具包编译
5. 确保 hook.so 和 server 程序使用相同的编译选项和 CUDA 版本

## 许可证

[待补充]

## 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进项目。
