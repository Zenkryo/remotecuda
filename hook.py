#!/usr/bin/env python3

import argparse
import os
import logging
import hashlib
import logging
from cxxheaderparser.simple import parse_file, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import Array, Pointer, Type, FunctionType, AnonymousName

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 默认的头文件和对应的 .so 文件
DEFAULT_H_SO_MAP = {
    "hidden_api.h": "/usr/local/cuda/lib64/stubs/libcudart.so",
    "/usr/local/cuda/include/cuda.h": "/usr/local/cuda/lib64/stubs/libcuda.so",
    "/usr/local/cuda/include/nvml.h": "/usr/local/cuda/lib64/stubs/libnvidia-ml.so",
    "/usr/local/cuda/include/cuda_runtime_api.h": "/usr/local/cuda/lib64/stubs/libcudart.so",
    # "/usr/local/cuda/include/cublas_api.h": "/usr/local/cuda/lib64/stubs/libcublas.so",
    # "/usr/local/cudnn/include/cudnn_graph.h": "/usr/local/cudnn/lib/libcudnn_graph.so",
    # "/usr/local/cudnn/include/cudnn_ops.h": "/usr/local/cudnn/lib/libcudnn_ops.so",
    # -------
    # "hidden_api.h": "/usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudart.so",
    # "/usr/local/cuda/include/cuda.h": "/usr/lib/x86_64-linux-gnu/libcuda.so",
    # "/usr/local/cuda/include/nvml.h": "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so",
    # "/usr/local/cuda/include/cuda_runtime_api.h": "/usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudart.so",
    # "/usr/local/cuda/include/cublas_api.h": "/usr/local/cuda-11.4/targets/x86_64-linux/lib/libcublas.so",
    # "/usr/include/cudnn_graph.h": "//usr/lib/x86_64-linux-gnu/libcudnn_graph.so",
    # "/usr/include/cudnn_ops.h": "/usr/lib/x86_64-linux-gnu/libcudnn_ops.so",
    # 可以继续添加其他默认的 .h 和 .so 文件对应关系
}

# 默认的包含目录
DEFAULT_INCLUDE_DIRS = ["/usr/local/cuda/include", "/usr/local/cudnn/include"]

# inline函数列表
INLINE_FUNCTIONS = [
    "cublasMigrateComputeType",
    "cublasGemmEx",
    "cublasGemmBatchedEx",
    "cublasGemmStridedBatchedEx",
]

# 手动实现的函数列表
MANUAL_FUNCTIONS = [
    "nvmlErrorString",
    "cuGetErrorString",
    "cuGetErrorName",
    "cudaGetErrorName",
    "cudaGetErrorString",
    "cudaMallocHost",
    "cudaHostAlloc",
    "cuMemAllocHost_v2",
    "cuMemHostAlloc",
    "cuMemCreate",
    "cuMemAddressReserve",
    "cuMemMap",
    "cudaMallocManaged",
    "cudaFreeHost",
    "cuMemFreeHost",
    "cudaFree",
    "cuMemRelease",
    "cudaMalloc",
    "cudaMallocPitch",
    "cudaMalloc3D",
    "cudaGetSymbolAddress",
    "cuMemAlloc_v2",
    "cuMemAllocPitch_v2",
    "cuMemAllocAsync",
    "cuMemAllocFromPoolAsync",
    "cuMemAllocManaged",
    "cuMemGetAddressRange_v2",
    "cuGraphicsResourceGetMappedPointer_v2",
    "cuTexRefGetAddress_v2",
    "cuGraphMemFreeNodeGetParams",
    "cuExternalMemoryGetMappedBuffer",
    "cuImportExternalMemory",
    "cuMemPoolImportPointer",
    "cuMemcpyBatchAsync",
    "cuLibraryGetManaged",
    "cuLibraryGetGlobal",
    "cuIpcOpenMemHandle_v2",
    "cuMemHostGetDevicePointer_v2",
    "cuModuleGetGlobal_v2",
    "cudaMemcpy",
    "cudaMemset",
    "cuGetProcAddress",
    "cudaLaunchKernel",
    "cudaMemcpyToSymbol",
    "cudaMemcpyFromSymbol",
    "__cudaInitModule",
    "__cudaPopCallConfiguration",
    "__cudaPushCallConfiguration",
    "__cudaRegisterFatBinary",
    "__cudaRegisterFatBinaryEnd",
    "__cudaRegisterFunction",
    "__cudaRegisterManagedVar",
    "__cudaRegisterVar",
    "__cudaUnregisterFatBinary",
]

# 隐藏类型
HIDDEN_TYPES = ["cudnnRuntimeTag_t"]


def generate_hook_api_h(output_dir, function_map):
    """
    生成 hook_api.h 文件，包含每个被 Hook 函数的唯一 key 和宏定义
    """
    hook_api_file = os.path.join(output_dir, "hook_api.h")

    with open(hook_api_file, "w") as f:
        f.write("#ifndef HOOK_API_H\n")
        f.write("#define HOOK_API_H\n\n")

        for header_file, functions in function_map.items():
            for function in functions:
                function_name = function.name.format()
                # 生成唯一的 hash key
                hash_key = int(hashlib.sha256(function_name.encode("utf-8")).hexdigest()[:8], 16)
                f.write(f"#define RPC_{function_name} 0x{hash_key:08X}\n")

        f.write("\n#endif // HOOK_API_H\n")


def generate_handle_server_h(output_dir, function_map):
    """
    生成 handle_server.h 文件，声明 RequestHandler 和 get_handler
    """
    handle_server_file = os.path.join(output_dir, "handle_server.h")

    with open(handle_server_file, "w") as f:
        f.write("#ifndef HOOK_SERVER_H\n")
        f.write("#define HOOK_SERVER_H\n\n")
        f.write("#include <cstdint>\n\n")
        f.write("typedef int (*RequestHandler)(void *);\n\n")
        f.write("RequestHandler get_handler(const uint32_t funcId);\n\n")
        # 生成每个被 Hook 函数的 handle 函数
        for header_file, functions in function_map.items():
            for function in functions:
                function_name = function.name.format()
                f.write(f"int handle_{function_name}(void *args);\n")
        f.write("#endif // HOOK_SERVER_H\n")


def generate_handle_server_cpp(output_dir, function_map):
    """
    生成 handle_server.cpp 文件，包含每个被 Hook 函数的 handle 函数和 handlerMap
    """
    handle_server_file = os.path.join(output_dir, "handle_server.cpp")

    with open(handle_server_file, "w") as f:
        # 包含 hook_api.h
        f.write("#include <unordered_map>\n")
        f.write("#include <iostream>\n\n")
        f.write('#include "hook_api.h"\n')
        f.write('#include "handle_server.h"\n\n')

        # 生成 handlerMap
        f.write("std::unordered_map<uint32_t, RequestHandler> handlerMap = {\n")
        for header_file, functions in function_map.items():
            for function in functions:
                function_name = function.name.format()
                # 使用 hook_api.h 中定义的 RPC_函数名 宏
                f.write(f"    {{RPC_{function_name}, handle_{function_name}}},\n")
        f.write("};\n\n")


def format_return_type_name(return_type):
    """
    格式化返回值类型名称
    - 如果返回值类型是指针，则返回值类型字符串后面加上 *
    - 如果返回值类型是指向指针的指针，则返回值类型字符串后面加上 **
    - 其他情况直接返回返回值类型字符串
    """
    if isinstance(return_type, Pointer):
        if isinstance(return_type.ptr_to, Type):
            return return_type.ptr_to.format() + " *"
        elif isinstance(return_type.ptr_to, Pointer):
            if isinstance(return_type.ptr_to.ptr_to, Type):
                return return_type.ptr_to.ptr_to.format() + " **"
    else:
        return return_type.format()


def generate_client_cpp(output_dir, function_map, so_files):
    """
    生成 hook_client.cpp 文件, 包含functionMape, 其为函数名和hook函数的映射
    """
    client_file = os.path.join(output_dir, "hook_client.cpp")

    with open(client_file, "w") as f:
        # 包含必要的头文件
        f.write("#include <iostream>\n")
        f.write("#include <cstdlib>\n")
        f.write("#include <dlfcn.h>\n")
        f.write("#include <unordered_map>\n")
        f.write("#include <string>\n\n")

        # 包含所有头文件
        for header_file in function_map.keys():
            if header_file.startswith("/"):
                f.write(f'#include "{os.path.basename(header_file)}"\n\n')
            else:
                f.write(f'#include "../{os.path.basename(header_file)}"\n\n')
        f.write("\n")

        # 写入自定义的 dlsym 实现
        f.write("// 使用 std::unordered_map 保存函数名和函数指针的映射\n")
        f.write("std::unordered_map<std::string, void *> functionMap = {\n")

        # 遍历所有头文件和函数，生成 map 初始化代码
        for header_file, functions in function_map.items():
            f.write(f"    // {os.path.basename(header_file)}\n")
            for function in functions:
                function_name = function.name.format()
                if function_name in INLINE_FUNCTIONS:
                    return_type = format_return_type_name(function.return_type)
                    # 函数参数列表
                    params = ", ".join([format_parameter(param) for param in function.parameters])
                    f.write(f'    {{"{function_name}", reinterpret_cast<void *>(static_cast<{return_type} (*)({params})>({function_name}))}},\n')
                else:
                    f.write(f'    {{"{function_name}", reinterpret_cast<void *>({function_name})}},\n')
        f.write("};\n\n")


def format_parameter(param):
    """
    格式化参数类型和参数名
    - 如果参数类型是 Array，移除参数类型字符串中的 []，并将 [] 加到参数名后
    """
    param_type = param.type
    param_name = param.name
    if isinstance(param_type, Array):
        # 如果参数类型是 Array，移除类型字符串中的 []
        base_type = param_type.array_of.format().replace("[]", "")
        return f"{base_type} {param_name}[]"
    elif isinstance(param_type, Pointer):
        if isinstance(param_type.ptr_to, Type):
            return f"{param_type.ptr_to.format()} *{param_name}"
        elif isinstance(param_type.ptr_to, Pointer):
            if isinstance(param_type.ptr_to.ptr_to, Type):
                return f"{param_type.ptr_to.ptr_to.format()} **{param_name}"
    else:
        # 其他类型直接格式化
        return f"{param_type.format()} {param_name}"


def getCharParamLength(function):
    """
    获取char *参数的长度参数名, 默认为32字节
    """
    for param in function.parameters:
        if isinstance(param.type, Type):
            if param.name in ["len", "max_size", "size", "length", "bufferSize"]:
                return param.name
        elif isinstance(param.type, Pointer):
            if param.name in ["bufferSize", "size", "length"]:
                return "*" + param.name
    return "32"


def getArrayLengthParam(function):
    """
    获取数组参数的长度参数名
    """
    for param in function.parameters:
        if isinstance(param.type, Type):
            if param.name in ["batchCount", "batchSize"]:
                return param.name
    return "0"


# 处理值类型参数
def handle_param_type(function, param, f, is_client=True, position=0):
    if is_client:
        # 客户端写入参数的值
        if position == 0:
            f.write(f"    rpc_write(client, &{param.name}, sizeof({param.name}));\n")
    else:
        if position == 0:
            # 服务器端定义同名变量
            f.write(f"    {param.type.format()} {param.name};\n")
            # 读取参数的值到上面定义的变量
            f.write(f"    rpc_read(client, &{param.name}, sizeof({param.name}));\n")
            # 返回参数名，用于调用真实函数
            return param.name


# 处理void *类型的参数
def handle_param_pvoid(function, param, f, is_client=True, position=0):
    if is_client:
        if position == 0:
            function_name = function.name.format()
            f.write(f"    void *_0{param.name} = getServerHostPtr({param.name});\n")
            if param.name == "dstHost":
                f.write(f"    rpc_write(client, &_0{param.name}, sizeof(_0{param.name}));\n")
                f.write(f"    rpc_read(client, {param.name}, ByteCount, true);\n")
            elif param.name == "value" and function_name in [
                "cuCoredumpGetAttribute",
                "cuCoredumpGetAttributeGlobal",
                "cuCoredumpSetAttribute",
                "cuCoredumpSetAttributeGlobal",
            ]:
                f.write(f"    rpc_write(client, &_0{param.name}, sizeof(_0{param.name}));\n")
                f.write(f"    rpc_read(client, {param.name}, *size, true);\n")
            else:
                f.write(f"    // PARAM void * {param.name}\n")
    else:
        if position == 0:
            f.write(f"    void *{param.name};\n")
            f.write(f"    // PARAM void * {param.name}\n")
            return param.name


# 处理const void *参数
def handle_param_pconstvoid(function, param, f, is_client=True, position=0):
    if is_client:
        if position == 0:
            if param.name == "srcHost":
                f.write(f"    void *_0{param.name} = getServerHostPtr((void *){param.name});\n")
                f.write(f"    rpc_write(client, &_0{param.name}, sizeof(_0{param.name}));\n")
                f.write(f"    rpc_write(client, {param.name}, ByteCount, true);\n")
            else:
                f.write(f"    rpc_write(client, &{param.name}, sizeof({param.name}), false);\n")
    else:
        if position == 0:
            if param.name == "srcHost":
                f.write(f"    void *{param.name};\n")
                f.write(f"    rpc_read(client, &{param.name}, sizeof({param.name}), false);\n")
                return param.name
            else:
                f.write(f"    void *{param.name};\n")
                f.write(f"    rpc_read(client, &{param.name}, sizeof({param.name}), false);\n")
                return param.name
        elif position == 1:
            if param.name == "srcHost":
                f.write(f"    if({param.name} == nullptr) {{\n")
                f.write(f"        read_one_now(client, &{param.name}, 0, true);\n")
                f.write(f"        buffers.insert({param.name});\n")
                f.write(f"    }}\n")
                f.write(f"    else {{\n")
                f.write(f"        read_one_now(client, {param.name}, ByteCount, true);\n")
                f.write(f"    }}\n")


# 处理char *参数
def handle_param_pchar(function, param, f, is_client=True, position=0):
    if is_client:
        if position == 0:
            # 现在都做输出参数处理，遇到例外再特殊处理
            len = getCharParamLength(function)
            f.write(f"    rpc_read(client, {param.name}, {len}, true);\n")
    else:
        if position == 0:
            # 服务器端定义一个局部变量来临时保存字符串
            f.write(f"    char {param.name}[1024];\n")
            return param.name
        elif position == 2:
            f.write(f"    rpc_write(client, {param.name}, strlen({param.name}) + 1, true);\n")


# 处理const char*参数
def handle_param_pconstchar(function, param, f, is_client=True, position=0):
    if is_client:
        if position == 0:
            # const char * 类型的参数必然是输入参数
            f.write(f"    rpc_write(client, {param.name}, strlen({param.name}) + 1, true);\n")
    else:
        if position == 0:
            # 服务器端定义一个局部变量来临时保存字符串
            f.write(f"    char *{param.name} = nullptr;\n")
            f.write(f"    rpc_read(client, &{param.name}, 0, true);\n")
            return param.name
        elif position == 1:
            f.write(f"    buffers.insert({param.name});\n")


# 处理type *参数
def handle_param_phidden(function, param, f, is_client=True, position=0):
    if is_client:
        if position == 0:
            # 对于指向隐藏类型的指针，将指针本身传递给服务器端
            f.write(f"    rpc_write(client, &{param.name}, sizeof({param.name}));\n")
    else:
        if position == 0:
            # 服务器端读取指针本身
            f.write(f"    {param.type.ptr_to.format()} *{param.name};\n")
            f.write(f"    rpc_read(client, &{param.name}, sizeof({param.name}));\n")
            return param.name


# 处理const type *参数
def handle_param_pconsttype(function, param, f, is_client=True, position=0):
    if is_client:
        if position == 0:
            f.write(f"    rpc_write(client, {param.name}, sizeof(*{param.name}));\n")
    else:
        if position == 0:
            # 移除参数类型字符串中开头的 const
            param_type_name = param.type.ptr_to.format()
            param_type_name = param_type_name[6:]
            f.write(f"    {param_type_name} {param.name};\n")
            f.write(f"    rpc_read(client, &{param.name}, sizeof({param.name}));\n")
            return "&" + param.name


# 处理输入类型的type *参数
def handle_param_ptype_in(function, param, f, is_client=True, position=0):
    if is_client:
        if position == 0:
            f.write(f"    rpc_write(client, {param.name}, sizeof(*{param.name}));\n")
    else:
        if position == 0:
            param_type_name = param.type.ptr_to.format()
            f.write(f"    {param_type_name} {param.name};\n")
            f.write(f"    rpc_read(client, &{param.name}, sizeof({param.name}));\n")
            return param.name


# 处理输出类型的
def handle_param_ptype_out(function, param, f, is_client=True, position=0):
    if is_client:
        if position == 0:
            f.write(f"    rpc_read(client, {param.name}, sizeof(*{param.name}));\n")
    else:
        if position == 0:
            param_type_name = param.type.ptr_to.format()
            f.write(f"    {param_type_name} {param.name};\n")
            return "&" + param.name
        elif position == 2:
            f.write(f"    rpc_write(client, &{param.name}, sizeof({param.name}));\n")


# 处理void **类型的参数
def handle_param_ppvoid(function, param, f, is_client=True, position=0):
    f.write(f"    // PARAM void **{param.name}\n")
    if is_client:
        pass
    else:
        if position == 0:
            f.write(f"    void *{param.name};\n")
            return "&" + param.name


# 处理const void **类型的参数
def handle_param_ppconstvoid(function, param, f, is_client=True, position=0):
    f.write(f"    // PARAM const void **{param.name}\n")
    if is_client:
        function_name = function.name.format()
        if position == 0:
            f.write(f"    rpc_read(client, {param.name}, sizeof(*{param.name}));\n")
    else:
        if position == 0:
            f.write(f"    const void *{param.name};\n")
            return "&" + param.name
        elif position == 2:
            f.write(f"    rpc_write(client, &{param.name}, sizeof({param.name}));\n")


# 处理char **类型的参数
def handle_param_ppchar(function, param, f, is_client=True, position=0):
    f.write(f"    // PARAM char **{param.name}\n")


# 处理const char **类型的参数
def handle_param_ppconstchar(function, param, f, is_client=True, position=0):
    f.write(f"    // PARAM const char **{param.name}\n")
    if is_client:
        function_name = function.name.format()
        if position == 0:
            f.write(f"    static char _{function_name}_{param.name}[1024];\n")
            f.write(f"    rpc_read(client, _{function_name}_{param.name}, 1024, true);\n")
        elif position == 1:
            f.write(f"    *{param.name} = _{function_name}_{param.name};\n")
    else:
        if position == 0:
            f.write(f"    const char *{param.name};\n")
            return "&" + param.name
        elif position == 2:
            f.write(f"    rpc_write(client, {param.name}, strlen({param.name}) + 1, true);\n")


# 处理const type **参数
def handle_param_ppconsttype(function, param, f, is_client=True, position=0):
    param_type_name = param.type.ptr_to.ptr_to.format()
    param_type_name = param_type_name[6:]
    f.write(f"    // PARAM const {param_type_name} **{param.name}\n")
    if is_client:
        function_name = function.name.format()
        if position == 0:
            # 定义一个静态变量
            f.write(f"    static {param_type_name} _{function_name}_{param.name};\n")
            f.write(f"    rpc_read(client, &_{function_name}_{param.name}, sizeof({param_type_name}));\n")
        elif position == 1:
            f.write(f"    *{param.name} = &_{function_name}_{param.name};\n")
    else:
        if position == 0:
            f.write(f"    const {param_type_name} *{param.name};\n")
            return "&" + param.name
        elif position == 2:
            f.write(f"    rpc_write(client, {param.name}, sizeof({param_type_name}));\n")


# 处理type * name[]参数
def handle_param_arrayptype(function, param, f, is_client=True, position=0):
    param_type_name = param.type.array_of.ptr_to.format()
    if param_type_name.startswith("const "):
        param_type_name = param_type_name[6:]
    len = getArrayLengthParam(function)
    if is_client:
        function_name = function.name.format()
        if position == 0:
            f.write(f"    rpc_write(client, {param.name}, sizeof({param_type_name} *)*{len}, true);\n")
    else:
        if position == 0:
            f.write(f"    {param_type_name} *{param.name} = nullptr;\n")
            f.write(f"    rpc_read(client, &{param.name}, 0, true);\n")
            return param.name  # TODO


def handle_param(function, param, f, is_client=True, position=0):
    """
    对某函数的某个参数进行处理
    -- is_client: 是客户端还是服务器端
    -- position==0:
        -- 如果是true
            -- 客户端: 在rpc_submit_request前根据参数类型进行处理
            -- 服务器端: 在rpc_prepare_response前根据参数类型进行处理
        -- 如果是false
            -- 客户端: 在rpc_submit_response后根据参数类型进行处理
            -- 服务器端: 在真实函数调用后根据参数类型进行处理
    """
    function_name = function.name.format()
    param_type = param.type
    if isinstance(param_type, Type):  # 值类型
        return handle_param_type(function, param, f, is_client, position)
    elif isinstance(param_type, Pointer):  # 指针类型
        if isinstance(param_type.ptr_to, Type):  # 值类型指针
            param_type_name = param_type.ptr_to.format()  # 指针指向的值的类型名
            if param_type_name == "void":  # void *
                return handle_param_pvoid(function, param, f, is_client, position)
            elif param_type_name == "const void":  # const void *
                return handle_param_pconstvoid(function, param, f, is_client, position)
            elif param_type_name == "char":  # char *
                return handle_param_pchar(function, param, f, is_client, position)
            elif param_type_name == "const char":  # const char *
                return handle_param_pconstchar(function, param, f, is_client, position)
            elif param_type_name in HIDDEN_TYPES:
                return handle_param_phidden(function, param, f, is_client, position)
            elif param_type.ptr_to.const:  # 指向const类型的指针
                return handle_param_pconsttype(function, param, f, is_client, position)
            elif param_type_name == "struct cudaGraphNodeParams":
                return handle_param_phidden(function, param, f, is_client, position)
            else:
                return handle_param_ptype_out(function, param, f, is_client, position)
        elif isinstance(param_type.ptr_to, Pointer):  # 指针的指针
            if isinstance(param_type.ptr_to.ptr_to, Type):  # 指向值类型的指针的指针
                if param_type.ptr_to.ptr_to.format() == "void":  # void **
                    return handle_param_ppvoid(function, param, f, is_client, position)
                elif param_type.ptr_to.ptr_to.format() == "const void":  # const void **
                    return handle_param_ppconstvoid(function, param, f, is_client, position)
                elif param_type.ptr_to.ptr_to.format() == "char":  # char **
                    return handle_param_ppchar(function, param, f, is_client, position)
                elif param_type.ptr_to.ptr_to.format() == "const char":  # const char **
                    return handle_param_ppconstchar(function, param, f, is_client, position)
                elif param_type.ptr_to.ptr_to.const:  # const type **
                    return handle_param_ppconsttype(function, param, f, is_client, position)
                else:
                    f.write(f'    std::cerr << "PARAM Not supported" << std::endl; // {param.name} \n')
                    f.write(f"    exit(1);\n")
            else:
                f.write(f'    std::cerr << "PARAM Not supported" << std::endl; // {param.name} \n')
                f.write(f"    exit(1);\n")
        else:
            f.write(f'    std::cerr << "PARAM Not supported" << std::endl; // {param.name} \n')
            f.write(f"    exit(1);\n")
    elif isinstance(param_type, Array):
        if isinstance(param_type.array_of, Pointer):
            if isinstance(param_type.array_of.ptr_to, Type):
                if not param_type.array_of.ptr_to.format().endswith("void"):
                    return handle_param_arrayptype(function, param, f, is_client, position)

        f.write(f'    std::cerr << "PARAM Not supported" << std::endl; // {param.name} \n')
        f.write(f"    exit(1);\n")
    else:
        f.write(f'    std::cerr << "PARAM Not supported" << std::endl; // {param.name} \n')
        f.write(f"    exit(1);\n")


def generate_hook_cpp(header_file, parsed_header, output_dir, function_map, so_file):
    """
    生成对应头文件的 .cpp 文件，包含 Hook 的函数实现
    """
    basename = os.path.splitext(os.path.basename(header_file))[0]
    output_file = os.path.join(output_dir, "hook_" + basename + ".cpp")

    with open(output_file, "w") as f:
        # 包含必要的头文件
        f.write("#include <iostream>\n")
        f.write("#include <unordered_map>\n")
        if header_file.startswith("/"):
            f.write(f'#include "{os.path.basename(header_file)}"\n\n')
        else:
            f.write(f'#include "../{os.path.basename(header_file)}"\n\n')
        f.write('#include "hook_api.h"\n')
        f.write('#include "../rpc.h"\n')

        # 声明 dlsym 函数指针
        f.write("extern void *(*real_dlsym)(void *, const char *);\n\n")
        f.write("void *getServerHostPtr(void *ptr);\n\n")

        f.write("void *get_so_handle(const std::string &so_file);\n")
        # 写入被 Hook 的函数实现
        if hasattr(parsed_header, "namespace") and hasattr(parsed_header.namespace, "functions"):
            function_map[header_file] = []
            for function in parsed_header.namespace.functions:
                if function.inline:
                    continue
                function_name = function.name.format()
                if function_name in INLINE_FUNCTIONS:
                    continue
                if function_name not in MANUAL_FUNCTIONS:
                    return_type = format_return_type_name(function.return_type)
                    # print("===========> ",function_name, " ",return_type)
                    # 函数参数列表
                    params = ", ".join([format_parameter(param) for param in function.parameters])
                    param_names = ", ".join([param.name for param in function.parameters])
                    if return_type.endswith("*"):
                        f.write(f'extern "C" {return_type}{function_name}({params}) {{\n')
                    else:
                        f.write(f'extern "C" {return_type} {function_name}({params}) {{\n')
                    f.write(f'    std::cout << "Hook: {function_name} called" << std::endl;\n')
                    # 如果函数的范围类型不是void，则需要定义一个变量来保存函数的返回值
                    if return_type == "const char *":
                        f.write(f"    char *_{function_name}_result = nullptr;\n")
                    elif return_type != "void":
                        if return_type.endswith("*"):
                            f.write(f"    {return_type}_result = nullptr;\n")
                        else:
                            f.write(f"    {return_type} _result;\n")
                    f.write(f"    RpcClient *client = rpc_get_client();\n")
                    f.write(f"    if(client == nullptr) {{\n")
                    f.write(f'        std::cerr << "Failed to get rpc client" << std::endl;\n')
                    f.write(f"        exit(1);\n")
                    f.write(f"    }}\n")
                    f.write(f"    rpc_prepare_request(client, RPC_{function_name});\n")
                    for param in function.parameters:
                        handle_param(function, param, f, True, 0)
                    if return_type == "const char *":
                        f.write(f"    rpc_read(client, &_{function_name}_result, 0, true);\n")
                    elif return_type != "void":
                        f.write(f"    rpc_read(client, &_result, sizeof(_result));\n")
                    f.write(f"    if(rpc_submit_request(client) != 0) {{\n")
                    f.write(f'        std::cerr << "Failed to submit request" << std::endl;\n')
                    f.write(f"        rpc_release_client(client);\n")
                    f.write(f"        exit(1);\n")
                    f.write(f"    }}\n")
                    for param in function.parameters:
                        handle_param(function, param, f, True, 1)
                    f.write(f"    rpc_free_client(client);\n")
                    if return_type == "const char *":
                        f.write(f"    return _{function_name}_result;\n")
                    elif return_type != "void":
                        f.write(f"    return _result;\n")
                    else:
                        f.write(f"    return;\n")
                    f.write("}\n\n")

                # 将函数名和函数原型添加到 function_map 中
                function_map[header_file].append(function)
    # 生成每个头文件对应的服务器端handle_.cpp文件
    output_file = os.path.join(output_dir, "handle_" + basename + ".cpp")
    with open(output_file, "w") as f:
        # 包含必要的头文件
        f.write("#include <iostream>\n")
        f.write("#include <unordered_map>\n")
        f.write('#include "hook_api.h"\n')
        f.write('#include "handle_server.h"\n')
        f.write('#include "../rpc.h"\n')
        if header_file.startswith("/"):
            f.write(f'#include "{os.path.basename(header_file)}"\n\n')
        else:
            f.write(f'#include "../{os.path.basename(header_file)}"\n\n')
        if hasattr(parsed_header, "namespace") and hasattr(parsed_header.namespace, "functions"):
            for function in parsed_header.namespace.functions:
                if function.inline:
                    continue
                function_name = function.name.format()
                if function_name in MANUAL_FUNCTIONS:
                    continue
                if function_name in INLINE_FUNCTIONS:
                    continue

                return_type = format_return_type_name(function.return_type)
                # 函数参数列表
                params = ", ".join([format_parameter(param) for param in function.parameters])
                param_names = ""

                f.write(f"int handle_{function_name}(void *args0) {{\n")
                f.write(f'    std::cout << "Handle function {function_name} called" << std::endl;\n')
                f.write(f"    int rtn = 0;\n")
                f.write(f"    std::set<void *> buffers;\n")
                f.write(f"    RpcClient *client = (RpcClient *)args0;\n")
                for param in function.parameters:  # 服务器端读取参数
                    if param_names == "":
                        p = handle_param(function, param, f, False, 0)
                        if p:
                            param_names = p
                    else:
                        p = handle_param(function, param, f, False, 0)
                        if p:
                            param_names = param_names + ", " + p
                if return_type.endswith("*"):
                    f.write(f"    {return_type}_result;\n")
                elif return_type != "void":
                    f.write(f"    {return_type} _result;\n")
                f.write(f"    if(rpc_prepare_response(client) != 0) {{\n")
                f.write(f'        std::cerr << "Failed to prepare response" << std::endl;\n')
                f.write("        rtn = 1;\n")
                f.write("        goto _RTN_;\n")
                f.write("    }\n")
                for param in function.parameters:  # 服务器端继续读取参数
                    handle_param(function, param, f, False, 1)
                if return_type.endswith("*"):
                    f.write(f"    _result = {function_name}({param_names});\n")
                elif return_type == "void":
                    f.write(f"    {function_name}({param_names});\n")
                else:
                    f.write(f"    _result = {function_name}({param_names});\n")
                for param in function.parameters:  # 服务器端写入返回参数
                    handle_param(function, param, f, False, 2)
                if return_type == "const char *":
                    f.write(f"    rpc_write(client, _result, strlen(_result) + 1, true);\n")
                elif return_type != "void":
                    f.write(f"    rpc_write(client, &_result, sizeof(_result));\n")
                f.write(f"    if(rpc_submit_response(client) != 0) {{\n")
                f.write(f'        std::cerr << "Failed to submit response" << std::endl;\n')
                f.write("        rtn = 1;\n")
                f.write(f"        goto _RTN_;\n")
                f.write("    }\n\n")

                f.write("_RTN_:\n")
                f.write(f"    for(auto it = buffers.begin(); it != buffers.end(); it++) {{\n")
                f.write(f"        ::free(*it);\n")
                f.write(f"    }}\n")
                for param in function.parameters:
                    handle_param(function, param, f, False, 3)  # 释放内存
                f.write("    return rtn;\n")
                f.write("}\n\n")


def generate_makefile(output_dir, hook_files, handle_files, include_dirs):
    """
    生成 Makefile 文件
    """
    makefile_path = os.path.join(output_dir, "Makefile")

    with open(makefile_path, "w") as f:
        # 定义编译器和基本编译参数
        f.write("CXX = g++\n")
        f.write("CXXFLAGS = -std=c++11 -fPIC -Wno-deprecated-declarations")
        for include_dir in include_dirs:
            f.write(f" -I{include_dir}")
        f.write(" -DCUBLASAPI= -DDEBUG\n")
        f.write("LDFLAGS = -ldl -lpthread\n\n")

        # 定义 hook.so 的编译参数和链接参数
        f.write("# Compilation flags for hook.so\n")
        f.write("HOOK_CXXFLAGS = $(CXXFLAGS)\n")
        f.write("HOOK_LDFLAGS = $(LDFLAGS) -shared\n\n")

        # 定义 server 的编译参数和链接参数
        f.write("# Compilation flags for server\n")
        f.write("SERVER_CXXFLAGS = $(CXXFLAGS)\n")
        f.write("SERVER_LDFLAGS = $(LDFLAGS) -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcuda -lnvidia-ml -lcublas -Wl,-rpath,/usr/local/cuda/lib64\n\n")

        # 定义 test 的编译参数和链接参数
        f.write("# Compilation flags for test\n")
        f.write("TEST_CXXFLAGS = $(CXXFLAGS)\n")
        f.write("TEST_LDFLAGS = $(LDFLAGS) -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcuda -lnvidia-ml -lcublas -Wl,-rpath,/usr/local/cuda/lib64\n\n")

        # 定义 hook.so 的源文件列表
        f.write("# Source files for hook.so\n")
        f.write("HOOK_SRCS = \\\n")
        for hook_file in hook_files:
            f.write(f"    {os.path.basename(hook_file)} \\\n")
        f.write(f"    ../manual_hook.cpp \\\n")
        f.write("    ../rpc.cpp\\\n")
        f.write(f"    ../client.cpp\n")
        f.write("\n")

        # 定义 server 的源文件列表
        f.write("# Source files for server\n")
        f.write("SERVER_SRCS = \\\n")
        for handle_file in handle_files:
            f.write(f"    {os.path.basename(handle_file)} \\\n")
        f.write("    handle_server.cpp\\\n")
        f.write("    ../manual_handle.cpp\\\n")
        f.write("    ../rpc.cpp\\\n")
        f.write("    ../rpc_server.cpp\\\n")
        f.write("    ../server.cpp\n\n")

        # 定义 test 的源文件列表
        f.write("# Source files for test\n")
        f.write("TEST_SRCS = \\\n")
        f.write("    ../test/test.cpp\n\n")

        # 定义目标文件列表
        f.write("HOOK_OBJS = $(HOOK_SRCS:.cpp=.o)\n")
        f.write("SERVER_OBJS = $(SERVER_SRCS:.cpp=.o)\n\n")
        f.write("TEST_OBJS = $(TEST_SRCS:.cpp=.o)\n\n")

        # 定义 all 目标
        f.write("all: hook.so server test\n\n")

        # 编译 hook.so
        f.write("hook.so: $(HOOK_OBJS)\n")
        f.write("\t$(CXX) $(HOOK_CXXFLAGS) -o $@ $^ $(HOOK_LDFLAGS)\n\n")

        # 编译 server
        f.write("server: $(SERVER_OBJS)\n")
        f.write("\t$(CXX) $(SERVER_CXXFLAGS) -o $@ $^ $(SERVER_LDFLAGS)\n\n")

        # 编译 test
        f.write("test: $(TEST_OBJS)\n")
        f.write("\t$(CXX) $(TEST_CXXFLAGS) -o $@ $^ $(TEST_LDFLAGS)\n\n")

        # 清理规则
        f.write("clean:\n")
        f.write("\trm -f $(HOOK_OBJS) $(SERVER_OBJS) $(TEST_OBJS) hook.so server test\n\n")

        # 伪目标
        f.write(".PHONY: all clean\n")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="解析头文件并生成对应的 .cpp 文件")
    parser.add_argument(
        "-I",
        "--input",
        nargs="+",
        default=list(DEFAULT_H_SO_MAP.keys()),
        help="输入的头文件路径（支持多个文件）",
    )
    parser.add_argument(
        "-S",
        "--so",
        nargs="+",
        default=list(DEFAULT_H_SO_MAP.values()),
        help="指定 .so 文件路径（可选，与 --input 一一对应）",
    )
    parser.add_argument(
        "-L",
        "--include",
        nargs="+",
        default=DEFAULT_INCLUDE_DIRS,
        help="头文件目录（支持多个目录）",
    )
    parser.add_argument("-O", "--output", default="", help="输出文件目录")
    args = parser.parse_args()

    # 检查 --input 和 --so 参数的数量是否一致
    if len(args.so) != len(args.input):
        logging.error("--input 和 --so 参数的数量必须一致")
        return

    # 检查args.input中的文件是否存在
    for input_file in args.input:
        if not os.path.exists(input_file):
            logging.error(f"未找到 {input_file} 文件")

    # 检查args.so中的文件是否存在
    for so_file in args.so:
        if not os.path.exists(so_file):
            logging.error(f"未找到 {so_file} 文件")
            return

    # 定义 gcc_args
    gcc_args = ["g++"]
    for include_dir in args.include:
        gcc_args.extend(["-I", include_dir])

    # 定义预处理器
    preprocessor = make_gcc_preprocessor(defines=["CUBLASAPI=", "_GCC_MAX_ALIGN_T="], gcc_args=gcc_args)

    # 配置 ParserOptions
    parser_options = ParserOptions(preprocessor=preprocessor)

    # 解析所有输入的头文件
    function_map = {}
    so_files = {}
    hook_files = []
    handle_files = []
    for index, input_file in enumerate(args.input):
        try:
            # 使用 cxxheaderparser 解析头文件，并启用预处理器
            parsed_header = parse_file(input_file, options=parser_options)
            logging.info(f"成功解析文件: {input_file}")

            # 获取对应的 .so 文件
            so_file = args.so[index]

            # 将 .so 文件添加到 map 中
            so_files[input_file] = so_file

            # 生成 Hook 的 .cpp 文件
            generate_hook_cpp(input_file, parsed_header, args.output, function_map, so_file)
            hook_files.append(
                os.path.join(
                    args.output,
                    "hook_" + os.path.splitext(os.path.basename(input_file))[0] + ".cpp",
                )
            )
            handle_files.append(
                os.path.join(
                    args.output,
                    "handle_" + os.path.splitext(os.path.basename(input_file))[0] + ".cpp",
                )
            )

        except Exception as e:
            logging.error(f"解析文件 {input_file} 时出错: {e}")

    # 生成 hook_client.cpp 文件
    generate_client_cpp(args.output, function_map, so_files)
    hook_files.append(os.path.join(args.output, "hook_client.cpp"))

    # 生成 hook_api.h
    generate_hook_api_h(args.output, function_map)

    # 生成 handle_server.h
    generate_handle_server_h(args.output, function_map)

    # 生成 hook_server.cpp
    generate_handle_server_cpp(args.output, function_map)

    # 生成 Makefile
    generate_makefile(args.output, hook_files, handle_files, args.include)


if __name__ == "__main__":
    main()
