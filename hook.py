#!/usr/bin/env python3

import argparse
import os
import logging
import hashlib
import logging
import random
import time
from cxxheaderparser.simple import parse_file, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import Array, Pointer, Type, FunctionType, AnonymousName

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 默认的头文件和对应的 .so 文件
DEFAULT_H_SO_MAP = {
    ##"hidden_api.h": "/usr/local/cuda/lib64/stubs/libcudart.so",
    ##"/usr/local/cuda/include/cuda.h": "/usr/local/cuda/lib64/stubs/libcuda.so",
    ##"/usr/local/cuda/include/nvml.h": "/usr/local/cuda/lib64/stubs/libnvidia-ml.so",
    ##"/usr/local/cuda/include/cuda_runtime_api.h": "/usr/local/cuda/lib64/stubs/libcudart.so",
    ##"/usr/local/cuda/include/cublas_api.h": "/usr/local/cuda/lib64/stubs/libcublas.so",
    # "/usr/local/cudnn/include/cudnn_graph.h": "/usr/local/cudnn/lib/libcudnn_graph.so",
    # "/usr/local/cudnn/include/cudnn_ops.h": "/usr/local/cudnn/lib/libcudnn_ops.so",
    # -------
    "hidden_api.h": "/usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudart.so",
    "/usr/local/cuda/include/cuda.h": "/usr/lib/x86_64-linux-gnu/libcuda.so",
    "/usr/local/cuda/include/nvml.h": "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so",
    "/usr/local/cuda/include/cuda_runtime_api.h": "/usr/local/cuda-11.4/targets/x86_64-linux/lib/libcudart.so",
    "/usr/local/cuda/include/cublas_api.h": "/usr/local/cuda-11.4/targets/x86_64-linux/lib/libcublas.so",
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
    # 客户端和服务器端内存同步
    "mem2server",
    "mem2client",
    # CUDA Runtime API
    "__cudaInitModule",
    "__cudaPopCallConfiguration",
    "__cudaPushCallConfiguration",
    "__cudaRegisterFatBinary",
    "__cudaRegisterFatBinaryEnd",
    "__cudaRegisterFunction",
    "__cudaRegisterManagedVar",
    "__cudaRegisterVar",
    "__cudaUnregisterFatBinary",
    "cudaFree",
    "cudaFreeHost",
    "cudaGetErrorName",
    "cudaGetErrorString",
    "cudaGetSymbolAddress",
    "cudaHostAlloc",
    "cudaHostRegister",
    "cudaHostUnregister",
    "cudaLaunchKernel",
    "cudaLaunchCooperativeKernel",
    "cudaMalloc",
    "cudaMalloc3D",
    "cudaMallocHost",
    "cudaMallocManaged",
    "cudaMallocPitch",
    "cudaMemRangeGetAttributes",
    # "cudaMemcpy",
    # "cudaMemcpyAsync",
    # "cudaMemcpyFromSymbol",
    # "cudaMemcpyToSymbol",
    # "cudaMemset",
    # "cudaMemsetAsync",
    # CUDA Driver API
    "cuExternalMemoryGetMappedBuffer",
    "cuGetErrorName",
    "cuGetErrorString",
    "cuGetProcAddress",
    "cuGraphMemFreeNodeGetParams",
    "cuGraphicsResourceGetMappedPointer_v2",
    "cuImportExternalMemory",
    "cuLaunchCooperativeKernel",
    "cuIpcOpenMemHandle_v2",
    "cuLibraryGetGlobal",
    "cuLibraryGetManaged",
    "cuMemAddressReserve",
    "cuMemAlloc_v2",
    # "cuMemAllocAsync",
    # "cuMemAllocFromPoolAsync",
    "cuMemAllocHost_v2",
    "cuMemAllocManaged",
    "cuMemAllocPitch_v2",
    "cuMemCreate",
    "cuMemFreeHost",
    "cuMemGetAddressRange_v2",
    "cuMemRangeGetAttributes",
    "cuMemHostAlloc",
    "cuMemHostGetDevicePointer_v2",
    "cuMemMap",
    "cuMemPoolImportPointer",
    "cuMemRelease",
    "cuPointerGetAttributes",
    # "cuMemcpyBatchAsync",
    "cuModuleGetGlobal_v2",
    "cuTexRefGetAddress_v2",
    # NVML
    "nvmlErrorString",
]


# 隐藏类型
HIDDEN_TYPES = [
    "cudnnRuntimeTag_t",
]
# "cudaStream_t",
# "cudaEvent_t",
# "cudaGraphicsResource_t",
# "cudaExternalMemory_t", "cudaExternalSemaphore_t", "cudaGraph_t", "cudaGraphNode_t", "cudaUserObject_t", "cudaFunction_t", "cudaMemPool_t", "cudaArray_t", "cudaArray_const_t", "cudaMipmappedArray_t", "cudaMipmappedArray_const_t"]


random.seed(time.time())
VERSION_KEY = random.randint(0, 0xFFFF)  # 定义一个16位的随机数作为版本密钥，每次生成的值都是随机的


def generate_hook_api_h(output_dir, function_map):
    """
    生成 hook_api.h 文件，包含每个被 Hook 函数的唯一 key 和宏定义
    """
    hook_api_file = os.path.join(output_dir, "hook_api.h")

    with open(hook_api_file, "w") as f:
        f.write("#ifndef HOOK_API_H\n")
        f.write("#define HOOK_API_H\n\n")
        f.write(f"#define VERSION_KEY 0x{VERSION_KEY:04X}\n\n")

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
        f.write('#include "../server.h"\n\n')
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
        f.write("void *getHookFunc(const char *symbol) {\n")
        f.write("    std::unordered_map<std::string, void *> functionMap = {\n")

        # 遍历所有头文件和函数，生成 map 初始化代码
        for header_file, functions in function_map.items():
            f.write(f"    // {os.path.basename(header_file)}\n")
            for function in functions:
                function_name = function.name.format()
                if function_name in INLINE_FUNCTIONS:
                    return_type = format_return_type_name(function.return_type)
                    # 函数参数列表
                    params = ", ".join([format_parameter(param) for param in function.parameters])
                    f.write(f'        {{"{function_name}", reinterpret_cast<void *>(static_cast<{return_type} (*)({params})>({function_name}))}},\n')
                else:
                    f.write(f'        {{"{function_name}", reinterpret_cast<void *>({function_name})}},\n')
        f.write("    };\n\n")
        f.write("    auto it = functionMap.find(symbol);\n")
        f.write("    if(it == functionMap.end()) {\n")
        f.write("        return nullptr;\n")
        f.write("    } else {\n")
        f.write("        return it->second;\n")
        f.write("    }\n")
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
        base_type = param_type.array_of.format().replace("[]", "").replace("* const", " *const")
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


def getArrayLengthParam(function, param):
    """
    获取数组参数的长度参数名
    """
    if "Group" in function.name.format():
        if isinstance(param.type.array_of, Pointer):
            return "sum_group((int *)group_size, group_count)"
        else:
            return "group_count"
    for _param in function.parameters:
        if isinstance(_param.type, Type):
            if _param.name in ["batchCount", "batchSize"]:
                return _param.name
    return "0"


# 获取指针指向内存的长度
def getPointerLength(function, param):
    param_name = param.name
    param_type = param.type.ptr_to.format()
    for _param in function.parameters:
        if _param.name == "ld" + param_name.lower() and _param.type.format() == "int":
            return _param.name
        if _param.type.format() == "cudaDataType" and (param_type == "void" or param_type == "const void"):
            if _param.name.endswith("type") or _param.name.endswith("Type"):
                pre = _param.name[:-4]
                if pre.find(param_name) != -1:
                    return "sizeofType(" + _param.name + ")"
    return "0"


# 处理值类型参数
def handle_param_type(function, param, f, is_client=True, position=0):
    if is_client:
        # 客户端写入参数的值
        if position == 1:
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
        len = calculate_pointer_sizes(function, param)
        if position == 0:
            f.write(f"    void *_0{param.name};\n")
            f.write(f"    mem2server(client, &_0{param.name}, (void *){param.name}, {len});\n")
        elif position == 1:
            f.write(f"    rpc_write(client, &_0{param.name}, sizeof(_0{param.name}));\n")
        elif position == 3:
            f.write(f"    mem2client(client, (void *){param.name}, {len});\n")
    else:
        if position == 0:
            f.write(f"    void *{param.name};\n")
            f.write(f"    rpc_read(client, &{param.name}, sizeof({param.name}));\n")
            return param.name


# 处理const void *参数
def handle_param_pconstvoid(function, param, f, is_client=True, position=0):
    if is_client:
        len = calculate_pointer_sizes(function, param)
        if position == 0:
            f.write(f"    void *_0{param.name};\n")
            f.write(f"    mem2server(client, &_0{param.name}, (void *){param.name}, {len});\n")
        elif position == 1:
            f.write(f"    rpc_write(client, &_0{param.name}, sizeof(_0{param.name}));\n")
        elif position == 3:
            f.write(f"    mem2client(client, (void *){param.name}, {len});\n")
    else:
        if position == 0:
            f.write(f"    void *{param.name};\n")
            f.write(f"    rpc_read(client, &{param.name}, sizeof({param.name}));\n")
            return param.name


# 处理char *参数
def handle_param_pchar(function, param, f, is_client=True, position=0):
    len = getCharParamLength(function)
    if is_client:
        if position == 1:
            # 现在都做输出参数处理，遇到例外再特殊处理
            f.write(f"    if({len} > 0) {{\n")
            f.write(f"        rpc_read(client, {param.name}, {len}, true);\n")
            f.write(f"    }}\n")
    else:
        if position == 0:
            # 服务器端定义一个局部变量来临时保存字符串
            f.write(f"    char {param.name}[1024];\n")
            return param.name
        elif position == 2:
            f.write(f"    if({len} > 0) {{\n")
            f.write(f"        rpc_write(client, {param.name}, strlen({param.name}) + 1, true);\n")
            f.write(f"    }}\n")


# 处理const char*参数
def handle_param_pconstchar(function, param, f, is_client=True, position=0):
    if is_client:
        if position == 1:
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
        if position == 1:
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
    param_type_name = param.type.ptr_to.format()
    if param_type_name.startswith("const "):
        param_type_name = param_type_name[6:]
    if is_client:
        len = calculate_pointer_sizes(function, param)
        if position == 0:
            f.write(f"    void *_0{param.name};\n")
            f.write(f"    mem2server(client, &_0{param.name}, (void *){param.name}, {len});\n")
        elif position == 1:
            f.write(f"    rpc_write(client, &_0{param.name}, sizeof(_0{param.name}));\n")
        elif position == 3:
            f.write(f"    mem2client(client, (void *){param.name}, {len});\n")
    else:
        if position == 0:
            f.write(f"    {param_type_name} *{param.name};\n")
            f.write(f"    rpc_read(client, &{param.name}, sizeof({param.name}));\n")
            return param.name


def handle_param_ptype(function, param, f, is_client=True, position=0):
    param_type_name = param.type.ptr_to.format()
    if is_client:
        len = calculate_pointer_sizes(function, param)
        if position == 0:
            f.write(f"    void *_0{param.name};\n")
            f.write(f"    mem2server(client, &_0{param.name}, (void *){param.name}, {len});\n")
        elif position == 1:
            f.write(f"    rpc_write(client, &_0{param.name}, sizeof(_0{param.name}));\n")
        elif position == 3:
            f.write(f"    mem2client(client, (void *){param.name}, {len});\n")
    else:
        if position == 0:
            f.write(f"    {param_type_name} *{param.name};\n")
            f.write(f"    rpc_read(client, &{param.name}, sizeof({param.name}));\n")
            return param.name


# 处理void **类型的参数
def handle_param_ppvoid(function, param, f, is_client=True, position=0):
    if is_client:
        if param.name in ["devPtr", "pDevice", "ptr", "funcPtr"]:
            if position == 1:
                f.write(f"    rpc_read(client, {param.name}, sizeof(void *));\n")
        else:
            f.write(f"    // PARAM void **{param.name}\n")
    else:
        if param.name in ["devPtr", "pDevice", "ptr", "funcPtr"]:
            if position == 2:
                f.write(f"    rpc_write(client, &{param.name}, sizeof({param.name}));\n")
        else:
            f.write(f"    // PARAM void **{param.name}\n")
        if position == 0:
            f.write(f"    void *{param.name};\n")
            return "&" + param.name


# 处理const void **类型的参数
def handle_param_ppconstvoid(function, param, f, is_client=True, position=0):
    if param.name not in ["ppExportTable"]:
        f.write(f"    // PARAM const void **{param.name}\n")
    if is_client:
        if position == 1:
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
        if position == 1:
            f.write(f"    static char _{function_name}_{param.name}[1024];\n")
            f.write(f"    rpc_read(client, _{function_name}_{param.name}, 1024, true);\n")
        elif position == 2:
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
    if is_client:
        function_name = function.name.format()
        if position == 1:
            # 定义一个静态变量
            f.write(f"    static {param_type_name} _{function_name}_{param.name};\n")
            f.write(f"    rpc_read(client, &_{function_name}_{param.name}, sizeof({param_type_name}));\n")
        elif position == 2:
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
    len = getArrayLengthParam(function, param)
    if is_client:
        if position == 1:
            f.write(f"    rpc_write(client, {param.name}, sizeof({param_type_name} *) * {len}, true);\n")
    else:
        if position == 0:
            f.write(f"    {param_type_name} *{param.name} = nullptr;\n")
            f.write(f"    rpc_read(client, &{param.name}, 0, true);\n")
            if param.type.array_of.const and param.type.array_of.ptr_to.const:
                return f"(const {param_type_name} *const *){param.name}"
            elif param.type.array_of.const:
                return f"({param_type_name} *const *){param.name}"
            elif param.type.array_of.ptr_to.const:
                return f"(const {param_type_name} **){param.name}"
            else:
                return param.name  # TODO


def handle_param_arraytype(function, param, f, is_client=True, position=0):
    param_type_name = param.type.array_of.format()
    if param_type_name.startswith("const "):
        param_type_name = param_type_name[6:]
    len = getArrayLengthParam(function, param)
    if is_client:
        function_name = function.name.format()
        if position == 1:
            f.write(f"    rpc_write(client, {param.name}, sizeof({param_type_name} *) * {len}, true);\n")
    else:
        if position == 0:
            f.write(f"    {param_type_name} *{param.name} = nullptr;\n")
            f.write(f"    rpc_read(client, &{param.name}, 0, true);\n")
            return param.name


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
                return handle_param_ptype(function, param, f, is_client, position)
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
                return handle_param_arrayptype(function, param, f, is_client, position)
        else:
            return handle_param_arraytype(function, param, f, is_client, position)

        f.write(f'    std::cerr << "PARAM Not supported" << std::endl; // {param.name} \n')
        f.write(f"    exit(1);\n")
    else:
        f.write(f'    std::cerr << "PARAM Not supported" << std::endl; // {param.name} \n')
        f.write(f"    exit(1);\n")


pointer_sizes = {
    "cublasCreate_v2": {"handle": "sizeof(*handle)"},
    "cublasGetVersion_v2": {"version": "sizeof(*version)"},
    "cublasGetProperty": {"value": "sizeof(*value)"},
    "cublasSetWorkspace_v2": {"workspace": "workspaceSizeInBytes"},
    "cublasGetStream_v2": {"streamId": "sizeof(*streamId)"},
    "cublasGetPointerMode_v2": {"mode": "sizeof(*mode)"},
    "cublasGetAtomicsMode": {"mode": "sizeof(*mode)"},
    "cublasGetMathMode": {"mode": "sizeof(*mode)"},
    "cublasGetSmCountTarget": {"smCountTarget": "sizeof(*smCountTarget)"},
    "cublasLoggerConfigure": {"logFileName": "strlen(logFileName)+1"},
    "cublasGetLoggerCallback": {"userCallback": "sizeof(*userCallback)"},
    "cublasSetVector": {"x": "n * elemSize", "devicePtr": "n * elemSize"},
    "cublasGetVector": {"x": "n * elemSize", "y": "n * elemSize"},
    "cublasSetMatrix": {"A": "rows * cols * elemSize", "B": "rows * cols * elemSize"},
    "cublasGetMatrix": {"A": "rows * cols * elemSize", "B": "rows * cols * elemSize"},
    "cublasSetVectorAsync": {"hostPtr": "n * elemSize", "devicePtr": "n * elemSize"},
    "cublasGetVectorAsync": {"devicePtr": "n * elemSize", "hostPtr": "n * elemSize"},
    "cublasSetMatrixAsync": {"A": "rows * cols * elemSize", "B": "rows * cols * elemSize"},
    "cublasGetMatrixAsync": {"A": "rows * cols * elemSize", "B": "rows * cols * elemSize"},
    "cublasXerbla": {"srName": "strlen(srName)+1"},
    "cublasNrm2Ex": {"x": "n * sizeofType(xType)", "result": "sizeofType(resultType)"},
    "cublasSnrm2_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasDnrm2_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasScnrm2_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasDznrm2_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasDotEx": {"x": "n * sizeofType(xType)", "y": "n * sizeofType(yType)", "result": "sizeofType(resultType)"},
    "cublasDotcEx": {"x": "n * sizeofType(xType)", "y": "n * sizeofType(yType)", "result": "sizeofType(resultType)"},
    "cublasSdot_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "result": "sizeof(*result)"},
    "cublasDdot_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "result": "sizeof(*result)"},
    "cublasCdotu_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "result": "sizeof(*result)"},
    "cublasCdotc_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "result": "sizeof(*result)"},
    "cublasZdotu_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "result": "sizeof(*result)"},
    "cublasZdotc_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "result": "sizeof(*result)"},
    "cublasScalEx": {"alpha": "sizeofType(alphaType)", "x": "n * sizeofType(xType)"},
    "cublasSscal_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)"},
    "cublasDscal_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)"},
    "cublasCscal_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)"},
    "cublasCsscal_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)"},
    "cublasZscal_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)"},
    "cublasZdscal_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)"},
    "cublasAxpyEx": {"alpha": "sizeofType(alphaType)", "x": "n * sizeofType(xType)", "y": "n * sizeofType(yType)"},
    "cublasSaxpy_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "y": "n * sizeof(*y)"},
    "cublasDaxpy_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "y": "n * sizeof(*y)"},
    "cublasCaxpy_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "y": "n * sizeof(*y)"},
    "cublasZaxpy_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "y": "n * sizeof(*y)"},
    "cublasCopyEx": {"x": "n * sizeofType(xType)", "y": "n * sizeofType(yType)"},
    "cublasScopy_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)"},
    "cublasDcopy_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)"},
    "cublasCcopy_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)"},
    "cublasZcopy_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)"},
    "cublasSswap_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)"},
    "cublasDswap_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)"},
    "cublasCswap_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)"},
    "cublasZswap_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)"},
    "cublasSwapEx": {"x": "n * sizeofType(xType)", "y": "n * sizeofType(yType)"},
    "cublasIsamax_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasIdamax_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasIcamax_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasIzamax_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasIamaxEx": {"x": "n * sizeofType(xType)", "result": "sizeof(*result)"},
    "cublasIsamin_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasIdamin_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasIcamin_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasIzamin_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasIaminEx": {"x": "n * sizeofType(xType)", "result": "sizeof(*result)"},
    "cublasAsumEx": {"x": "n * sizeofType(xType)", "result": "sizeofType(resultType)"},
    "cublasSasum_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasDasum_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasScasum_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasDzasum_v2": {"x": "n * sizeof(*x)", "result": "sizeof(*result)"},
    "cublasSrot_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "c": "sizeof(*c)", "s": "sizeof(*s)"},
    "cublasDrot_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "c": "sizeof(*c)", "s": "sizeof(*s)"},
    "cublasCrot_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "c": "sizeof(*c)", "s": "sizeof(*s)"},
    "cublasCsrot_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "c": "sizeof(*c)", "s": "sizeof(*s)"},
    "cublasZrot_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "c": "sizeof(*c)", "s": "sizeof(*s)"},
    "cublasZdrot_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "c": "sizeof(*c)", "s": "sizeof(*s)"},
    "cublasRotEx": {"x": "n * sizeofType(xType)", "y": "n * sizeofType(yType)", "c": "sizeofType(csType)", "s": "sizeofType(csType)"},
    "cublasSrotg_v2": {"a": "sizeof(*a)", "b": "sizeof(*b)", "c": "sizeof(*c)", "s": "sizeof(*s)"},
    "cublasDrotg_v2": {"a": "sizeof(*a)", "b": "sizeof(*b)", "c": "sizeof(*c)", "s": "sizeof(*s)"},
    "cublasCrotg_v2": {"a": "sizeof(*a)", "b": "sizeof(*b)", "c": "sizeof(*c)", "s": "sizeof(*s)"},
    "cublasZrotg_v2": {"a": "sizeof(*a)", "b": "sizeof(*b)", "c": "sizeof(*c)", "s": "sizeof(*s)"},
    "cublasRotgEx": {"a": "sizeofType(abType)", "b": "sizeofType(abType)", "c": "sizeofType(csType)", "s": "sizeofType(csType)"},
    "cublasSrotm_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "param": "5 * sizeof(*param)"},
    "cublasDrotm_v2": {"x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "param": "5 * sizeof(*param)"},
    "cublasRotmEx": {"x": "n * sizeofType(xType)", "y": "n * sizeofType(yType)", "param": "5 * sizeofType(paramType)"},
    "cublasSrotmg_v2": {"d1": "sizeof(*d1)", "d2": "sizeof(*d2)", "x1": "sizeof(*x1)", "y1": "sizeof(*y1)", "param": "5 * sizeof(*param)"},
    "cublasDrotmg_v2": {"d1": "sizeof(*d1)", "d2": "sizeof(*d2)", "x1": "sizeof(*x1)", "y1": "sizeof(*y1)", "param": "5 * sizeof(*param)"},
    "cublasRotmgEx": {"d1": "sizeofType(d1Type)", "d2": "sizeofType(d2Type)", "x1": "sizeofType(x1Type)", "y1": "sizeofType(y1Type)", "param": "5 * sizeofType(paramType)"},
    "cublasSgemv_v2": {"alpha": "sizeof(*alpha)", "A": "m * n * sizeof(*A)", "x": "(trans == CUBLAS_OP_N ? n : m) * sizeof(*x)", "beta": "sizeof(*beta)", "y": "(trans == CUBLAS_OP_N ? m : n) * sizeof(*y)"},
    "cublasDgemv_v2": {"alpha": "sizeof(*alpha)", "A": "m * n * sizeof(*A)", "x": "(trans == CUBLAS_OP_N ? n : m) * sizeof(*x)", "beta": "sizeof(*beta)", "y": "(trans == CUBLAS_OP_N ? m : n) * sizeof(*y)"},
    "cublasCgemv_v2": {"alpha": "sizeof(*alpha)", "A": "m * n * sizeof(*A)", "x": "(trans == CUBLAS_OP_N ? n : m) * sizeof(*x)", "beta": "sizeof(*beta)", "y": "(trans == CUBLAS_OP_N ? m : n) * sizeof(*y)"},
    "cublasZgemv_v2": {"alpha": "sizeof(*alpha)", "A": "m * n * sizeof(*A)", "x": "(trans == CUBLAS_OP_N ? n : m) * sizeof(*x)", "beta": "sizeof(*beta)", "y": "(trans == CUBLAS_OP_N ? m : n) * sizeof(*y)"},
    "cublasSgbmv_v2": {"alpha": "sizeof(*alpha)", "A": "m * n * sizeof(*A)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "m * sizeof(*y)"},
    "cublasDgbmv_v2": {"alpha": "sizeof(*alpha)", "A": "m * n * sizeof(*A)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "m * sizeof(*y)"},
    "cublasCgbmv_v2": {"alpha": "sizeof(*alpha)", "A": "m * n * sizeof(*A)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "m * sizeof(*y)"},
    "cublasZgbmv_v2": {"alpha": "sizeof(*alpha)", "A": "m * n * sizeof(*A)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "m * sizeof(*y)"},
    "cublasStrmv_v2": {"A": "n * n * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasDtrmv_v2": {"A": "n * n * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasCtrmv_v2": {"A": "n * n * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasZtrmv_v2": {"A": "n * n * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasStbmv_v2": {"A": "n * k * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasDtbmv_v2": {"A": "n * k * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasCtbmv_v2": {"A": "n * k * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasZtbmv_v2": {"A": "n * k * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasStpmv_v2": {"AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "x": "n * sizeof(*x)"},
    "cublasDtpmv_v2": {"AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "x": "n * sizeof(*x)"},
    "cublasCtpmv_v2": {"AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "x": "n * sizeof(*x)"},
    "cublasZtpmv_v2": {"AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "x": "n * sizeof(*x)"},
    "cublasStrsv_v2": {"A": "n * n * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasDtrsv_v2": {"A": "n * n * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasCtrsv_v2": {"A": "n * n * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasZtrsv_v2": {"A": "n * n * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasStpsv_v2": {"AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "x": "n * sizeof(*x)"},
    "cublasDtpsv_v2": {"AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "x": "n * sizeof(*x)"},
    "cublasCtpsv_v2": {"AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "x": "n * sizeof(*x)"},
    "cublasZtpsv_v2": {"AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "x": "n * sizeof(*x)"},
    "cublasStbsv_v2": {"A": "n * k * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasDtbsv_v2": {"A": "n * k * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasCtbsv_v2": {"A": "n * k * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasZtbsv_v2": {"A": "n * k * sizeof(*A)", "x": "n * sizeof(*x)"},
    "cublasSsymv_v2": {"alpha": "sizeof(*alpha)", "A": "n * n * sizeof(*A)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "n * sizeof(*y)"},
    "cublasDsymv_v2": {"alpha": "sizeof(*alpha)", "A": "n * n * sizeof(*A)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "n * sizeof(*y)"},
    "cublasCsymv_v2": {"alpha": "sizeof(*alpha)", "A": "n * n * sizeof(*A)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "n * sizeof(*y)"},
    "cublasZsymv_v2": {"alpha": "sizeof(*alpha)", "A": "n * n * sizeof(*A)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "n * sizeof(*y)"},
    "cublasChemv_v2": {"alpha": "sizeof(*alpha)", "A": "n * n * sizeof(*A)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "n * sizeof(*y)"},
    "cublasZhemv_v2": {"alpha": "sizeof(*alpha)", "A": "n * n * sizeof(*A)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "n * sizeof(*y)"},
    "cublasSsbmv_v2": {"alpha": "sizeof(*alpha)", "A": "n * k * sizeof(*A)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "n * sizeof(*y)"},
    "cublasDsbmv_v2": {"alpha": "sizeof(*alpha)", "A": "n * k * sizeof(*A)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "n * sizeof(*y)"},
    "cublasChbmv_v2": {"alpha": "sizeof(*alpha)", "A": "n * k * sizeof(*A)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "n * sizeof(*y)"},
    "cublasZhbmv_v2": {"alpha": "sizeof(*alpha)", "A": "n * k * sizeof(*A)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "n * sizeof(*y)"},
    "cublasSspmv_v2": {"alpha": "sizeof(*alpha)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "n * sizeof(*y)"},
    "cublasDspmv_v2": {"alpha": "sizeof(*alpha)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "n * sizeof(*y)"},
    "cublasChpmv_v2": {"alpha": "sizeof(*alpha)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "n * sizeof(*y)"},
    "cublasZhpmv_v2": {"alpha": "sizeof(*alpha)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "x": "n * sizeof(*x)", "beta": "sizeof(*beta)", "y": "n * sizeof(*y)"},
    "cublasSger_v2": {"alpha": "sizeof(*alpha)", "x": "m * sizeof(*x)", "y": "n * sizeof(*y)", "A": "m * n * sizeof(*A)"},
    "cublasDger_v2": {"alpha": "sizeof(*alpha)", "x": "m * sizeof(*x)", "y": "n * sizeof(*y)", "A": "m * n * sizeof(*A)"},
    "cublasCgeru_v2": {"alpha": "sizeof(*alpha)", "x": "m * sizeof(*x)", "y": "n * sizeof(*y)", "A": "m * n * sizeof(*A)"},
    "cublasCgerc_v2": {"alpha": "sizeof(*alpha)", "x": "m * sizeof(*x)", "y": "n * sizeof(*y)", "A": "m * n * sizeof(*A)"},
    "cublasZgeru_v2": {"alpha": "sizeof(*alpha)", "x": "m * sizeof(*x)", "y": "n * sizeof(*y)", "A": "m * n * sizeof(*A)"},
    "cublasZgerc_v2": {"alpha": "sizeof(*alpha)", "x": "m * sizeof(*x)", "y": "n * sizeof(*y)", "A": "m * n * sizeof(*A)"},
    "cublasSsyr_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "A": "n * n * sizeof(*A)"},
    "cublasDsyr_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "A": "n * n * sizeof(*A)"},
    "cublasCsyr_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "A": "n * n * sizeof(*A)"},
    "cublasZsyr_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "A": "n * n * sizeof(*A)"},
    "cublasCher_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "A": "n * n * sizeof(*A)"},
    "cublasZher_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "A": "n * n * sizeof(*A)"},
    "cublasSspr_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)"},
    "cublasDspr_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)"},
    "cublasChpr_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)"},
    "cublasZhpr_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)"},
    "cublasSsyr2_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "A": "n * n * sizeof(*A)"},
    "cublasDsyr2_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "A": "n * n * sizeof(*A)"},
    "cublasCsyr2_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "A": "n * n * sizeof(*A)"},
    "cublasZsyr2_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "A": "n * n * sizeof(*A)"},
    "cublasCher2_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "A": "n * n * sizeof(*A)"},
    "cublasZher2_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "A": "n * n * sizeof(*A)"},
    "cublasSspr2_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)"},
    "cublasDspr2_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)"},
    "cublasChpr2_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)"},
    "cublasZhpr2_v2": {"alpha": "sizeof(*alpha)", "x": "n * sizeof(*x)", "y": "n * sizeof(*y)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)"},
    "cublasSgemm_v2": {"alpha": "sizeof(*alpha)", "A": "transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A)", "B": "transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasDgemm_v2": {"alpha": "sizeof(*alpha)", "A": "transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A)", "B": "transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasCgemm_v2": {"alpha": "sizeof(*alpha)", "A": "transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A)", "B": "transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasCgemm3m": {"alpha": "sizeof(*alpha)", "A": "transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A)", "B": "transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasCgemm3mEx": {"alpha": "sizeof(*alpha)", "A": "transa == CUBLAS_OP_N ? m * k : k * m * sizeofType(Atype)", "B": "transb == CUBLAS_OP_N ? k * n : n * k * sizeofType(Btype)", "beta": "sizeof(*beta)", "C": "m * n * sizeofType(Ctype)"},
    "cublasZgemm_v2": {"alpha": "sizeof(*alpha)", "A": "transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A)", "B": "transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasZgemm3m": {"alpha": "sizeof(*alpha)", "A": "transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A)", "B": "transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasHgemm": {"alpha": "sizeof(*alpha)", "A": "transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A)", "B": "transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasSgemmEx": {"alpha": "sizeof(*alpha)", "A": "transa == CUBLAS_OP_N ? m * k : k * m * sizeofType(Atype)", "B": "transb == CUBLAS_OP_N ? k * n : n * k * sizeofType(Btype)", "beta": "sizeof(*beta)", "C": "m * n * sizeofType(Ctype)"},
    "cublasCgemmEx": {"alpha": "sizeof(*alpha)", "A": "transa == CUBLAS_OP_N ? m * k : k * m * sizeofType(Atype)", "B": "transb == CUBLAS_OP_N ? k * n : n * k * sizeofType(Btype)", "beta": "sizeof(*beta)", "C": "m * n * sizeofType(Ctype)"},
    "cublasUint8gemmBias": {"A": "transa == CUBLAS_OP_N ? m * k : k * m * sizeof(*A)", "B": "transb == CUBLAS_OP_N ? k * n : n * k * sizeof(*B)", "C": "m * n * sizeof(*C)"},
    "cublasSsyrk_v2": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasDsyrk_v2": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasCsyrk_v2": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasZsyrk_v2": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasCsyrkEx": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeofType(Atype)", "beta": "sizeof(*beta)", "C": "n * n * sizeofType(Ctype)"},
    "cublasCsyrk3mEx": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeofType(Atype)", "beta": "sizeof(*beta)", "C": "n * n * sizeofType(Ctype)"},
    "cublasCherk_v2": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasZherk_v2": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasCherkEx": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeofType(Atype)", "beta": "sizeof(*beta)", "C": "n * n * sizeofType(Ctype)"},
    "cublasCherk3mEx": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeofType(Atype)", "beta": "sizeof(*beta)", "C": "n * n * sizeofType(Ctype)"},
    "cublasSsyr2k_v2": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "B": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasDsyr2k_v2": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "B": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasCsyr2k_v2": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "B": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasZsyr2k_v2": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "B": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasCher2k_v2": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "B": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasZher2k_v2": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "B": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasSsyrkx": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "B": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasDsyrkx": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "B": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasCsyrkx": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "B": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasZsyrkx": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "B": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasCherkx": {"alpha": "sizeof(*alpha)", "A": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*A)", "B": "trans == CUBLAS_OP_N ? n * k : k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasZherkx": {"alpha": "sizeof(*alpha)", "A": "n * k * sizeof(*A)", "B": "n * k * sizeof(*B)", "beta": "sizeof(*beta)", "C": "n * n * sizeof(*C)"},
    "cublasSsymm_v2": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A)", "B": "m * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasDsymm_v2": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A)", "B": "m * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasCsymm_v2": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A)", "B": "m * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasZsymm_v2": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A)", "B": "m * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasChemm_v2": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A)", "B": "m * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasZhemm_v2": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A)", "B": "m * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasStrsm_v2": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A)", "B": "m * n * sizeof(*B)"},
    "cublasDtrsm_v2": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A)", "B": "m * n * sizeof(*B)"},
    "cublasCtrsm_v2": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A)", "B": "m * n * sizeof(*B)"},
    "cublasZtrsm_v2": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A)", "B": "m * n * sizeof(*B)"},
    "cublasStrmm_v2": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A)", "B": "m * n * sizeof(*B)", "C": "m * n * sizeof(*C)"},
    "cublasDtrmm_v2": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A)", "B": "m * n * sizeof(*B)", "C": "m * n * sizeof(*C)"},
    "cublasCtrmm_v2": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A)", "B": "m * n * sizeof(*B)", "C": "m * n * sizeof(*C)"},
    "cublasZtrmm_v2": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(*A)", "B": "m * n * sizeof(*B)", "C": "m * n * sizeof(*C)"},
    "cublasHgemmBatched": {"alpha": "sizeof(*alpha)", "Aarray": "m * k * sizeof(**Aarray)", "Barray": "k * n * sizeof(**Barray)", "beta": "sizeof(*beta)", "Carray": "m * n * sizeof(**Carray)"},
    "cublasSgemmBatched": {"alpha": "sizeof(*alpha)", "Aarray": "m * k * sizeof(**Aarray)", "Barray": "k * n * sizeof(**Barray)", "beta": "sizeof(*beta)", "Carray": "m * n * sizeof(**Carray)"},
    "cublasDgemmBatched": {"alpha": "sizeof(*alpha)", "Aarray": "m * k * sizeof(**Aarray)", "Barray": "k * n * sizeof(**Barray)", "beta": "sizeof(*beta)", "Carray": "m * n * sizeof(**Carray)"},
    "cublasCgemmBatched": {"alpha": "sizeof(*alpha)", "Aarray": "m * k * sizeof(**Aarray)", "Barray": "k * n * sizeof(**Barray)", "beta": "sizeof(*beta)", "Carray": "m * n * sizeof(**Carray)"},
    "cublasCgemm3mBatched": {"alpha": "sizeof(*alpha)", "Aarray": "m * k * sizeof(**Aarray)", "Barray": "k * n * sizeof(**Barray)", "beta": "sizeof(*beta)", "Carray": "m * n * sizeof(**Carray)"},
    "cublasZgemmBatched": {"alpha": "sizeof(*alpha)", "Aarray": "m * k * sizeof(**Aarray)", "Barray": "k * n * sizeof(**Barray)", "beta": "sizeof(*beta)", "Carray": "m * n * sizeof(**Carray)"},
    "cublasSgemmStridedBatched": {"alpha": "sizeof(*alpha)", "A": "m * k * sizeof(*A)", "B": "k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasDgemmStridedBatched": {"alpha": "sizeof(*alpha)", "A": "m * k * sizeof(*A)", "B": "k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasCgemmStridedBatched": {"alpha": "sizeof(*alpha)", "A": "m * k * sizeof(*A)", "B": "k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasCgemm3mStridedBatched": {"alpha": "sizeof(*alpha)", "A": "m * k * sizeof(*A)", "B": "k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasZgemmStridedBatched": {"alpha": "sizeof(*alpha)", "A": "m * k * sizeof(*A)", "B": "k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasHgemmStridedBatched": {"alpha": "sizeof(*alpha)", "A": "m * k * sizeof(*A)", "B": "k * n * sizeof(*B)", "beta": "sizeof(*beta)", "C": "m * n * sizeof(*C)"},
    "cublasSgeam": {"alpha": "sizeof(*alpha)", "A": "m * n * sizeof(*A)", "beta": "sizeof(*beta)", "B": "m * n * sizeof(*B)", "C": "m * n * sizeof(*C)"},
    "cublasDgeam": {"alpha": "sizeof(*alpha)", "A": "m * n * sizeof(*A)", "beta": "sizeof(*beta)", "B": "m * n * sizeof(*B)", "C": "m * n * sizeof(*C)"},
    "cublasCgeam": {"alpha": "sizeof(*alpha)", "A": "m * n * sizeof(*A)", "beta": "sizeof(*beta)", "B": "m * n * sizeof(*B)", "C": "m * n * sizeof(*C)"},
    "cublasZgeam": {"alpha": "sizeof(*alpha)", "A": "m * n * sizeof(*A)", "beta": "sizeof(*beta)", "B": "m * n * sizeof(*B)", "C": "m * n * sizeof(*C)"},
    "cublasSgetrfBatched": {"A": "n * n * sizeof(**A)", "P": "n * sizeof(*P)", "info": "batchSize * sizeof(*info)"},
    "cublasDgetrfBatched": {"A": "n * n * sizeof(**A)", "P": "n * sizeof(*P)", "info": "batchSize * sizeof(*info)"},
    "cublasCgetrfBatched": {"A": "n * n * sizeof(**A)", "P": "n * sizeof(*P)", "info": "batchSize * sizeof(*info)"},
    "cublasZgetrfBatched": {"A": "n * n * sizeof(**A)", "P": "n * sizeof(*P)", "info": "batchSize * sizeof(*info)"},
    "cublasSgetriBatched": {"A": "n * n * sizeof(**A)", "P": "n * sizeof(*P)", "C": "n * n * sizeof(**C)", "info": "batchSize * sizeof(*info)"},
    "cublasDgetriBatched": {"A": "n * n * sizeof(**A)", "P": "n * sizeof(*P)", "C": "n * n * sizeof(**C)", "info": "batchSize * sizeof(*info)"},
    "cublasCgetriBatched": {"A": "n * n * sizeof(**A)", "P": "n * sizeof(*P)", "C": "n * n * sizeof(**C)", "info": "batchSize * sizeof(*info)"},
    "cublasZgetriBatched": {"A": "n * n * sizeof(**A)", "P": "n * sizeof(*P)", "C": "n * n * sizeof(**C)", "info": "batchSize * sizeof(*info)"},
    "cublasSgetrsBatched": {"Aarray": "n * n * sizeof(**Aarray)", "devIpiv": "n * sizeof(*devIpiv)", "Barray": "n * nrhs * sizeof(**Barray)", "info": "batchSize * sizeof(*info)"},
    "cublasDgetrsBatched": {"Aarray": "n * n * sizeof(**Aarray)", "devIpiv": "n * sizeof(*devIpiv)", "Barray": "n * nrhs * sizeof(**Barray)", "info": "batchSize * sizeof(*info)"},
    "cublasCgetrsBatched": {"Aarray": "n * n * sizeof(**Aarray)", "devIpiv": "n * sizeof(*devIpiv)", "Barray": "n * nrhs * sizeof(**Barray)", "info": "batchSize * sizeof(*info)"},
    "cublasZgetrsBatched": {"Aarray": "n * n * sizeof(**Aarray)", "devIpiv": "n * sizeof(*devIpiv)", "Barray": "n * nrhs * sizeof(**Barray)", "info": "batchSize * sizeof(*info)"},
    "cublasStrsmBatched": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(**A)", "B": "m * n * sizeof(**B)"},
    "cublasDtrsmBatched": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(**A)", "B": "m * n * sizeof(**B)"},
    "cublasCtrsmBatched": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(**A)", "B": "m * n * sizeof(**B)"},
    "cublasZtrsmBatched": {"alpha": "sizeof(*alpha)", "A": "(side == CUBLAS_SIDE_LEFT ? m * m : n * n) * sizeof(**A)", "B": "m * n * sizeof(**B)"},
    "cublasSmatinvBatched": {"A": "n * n * sizeof(**A)", "Ainv": "n * n * sizeof(**Ainv)", "info": "batchSize * sizeof(*info)"},
    "cublasDmatinvBatched": {"A": "n * n * sizeof(**A)", "Ainv": "n * n * sizeof(**Ainv)", "info": "batchSize * sizeof(*info)"},
    "cublasCmatinvBatched": {"A": "n * n * sizeof(**A)", "Ainv": "n * n * sizeof(**Ainv)", "info": "batchSize * sizeof(*info)"},
    "cublasZmatinvBatched": {"A": "n * n * sizeof(**A)", "Ainv": "n * n * sizeof(**Ainv)", "info": "batchSize * sizeof(*info)"},
    "cublasSgeqrfBatched": {"Aarray": "m * n * sizeof(**Aarray)", "TauArray": "(m < n ? m : n) * sizeof(**TauArray)", "info": "batchSize * sizeof(*info)"},
    "cublasDgeqrfBatched": {"Aarray": "m * n * sizeof(**Aarray)", "TauArray": "(m < n ? m : n) * sizeof(**TauArray)", "info": "batchSize * sizeof(*info)"},
    "cublasCgeqrfBatched": {"Aarray": "m * n * sizeof(**Aarray)", "TauArray": "(m < n ? m : n) * sizeof(**TauArray)", "info": "batchSize * sizeof(*info)"},
    "cublasZgeqrfBatched": {"Aarray": "m * n * sizeof(**Aarray)", "TauArray": "(m < n ? m : n) * sizeof(**TauArray)", "info": "batchSize * sizeof(*info)"},
    "cublasSgelsBatched": {"Aarray": "m * n * sizeof(**Aarray)", "Carray": "m * nrhs * sizeof(**Carray)", "info": "batchSize * sizeof(*info)", "devInfoArray": "batchSize * sizeof(*devInfoArray)"},
    "cublasDgelsBatched": {"Aarray": "m * n * sizeof(**Aarray)", "Carray": "m * nrhs * sizeof(**Carray)", "info": "batchSize * sizeof(*info)", "devInfoArray": "batchSize * sizeof(*devInfoArray)"},
    "cublasCgelsBatched": {"Aarray": "m * n * sizeof(**Aarray)", "Carray": "m * nrhs * sizeof(**Carray)", "info": "batchSize * sizeof(*info)", "devInfoArray": "batchSize * sizeof(*devInfoArray)"},
    "cublasZgelsBatched": {"Aarray": "m * n * sizeof(**Aarray)", "Carray": "m * nrhs * sizeof(**Carray)", "info": "batchSize * sizeof(*info)", "devInfoArray": "batchSize * sizeof(*devInfoArray)"},
    "cublasSdgmm": {"A": "m * n * sizeof(*A)", "x": "(mode == CUBLAS_SIDE_LEFT ? m : n) * sizeof(*x)", "C": "m * n * sizeof(*C)"},
    "cublasDdgmm": {"A": "m * n * sizeof(*A)", "x": "(mode == CUBLAS_SIDE_LEFT ? m : n) * sizeof(*x)", "C": "m * n * sizeof(*C)"},
    "cublasCdgmm": {"A": "m * n * sizeof(*A)", "x": "(mode == CUBLAS_SIDE_LEFT ? m : n) * sizeof(*x)", "C": "m * n * sizeof(*C)"},
    "cublasZdgmm": {"A": "m * n * sizeof(*A)", "x": "(mode == CUBLAS_SIDE_LEFT ? m : n) * sizeof(*x)", "C": "m * n * sizeof(*C)"},
    "cublasStpttr": {"AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "A": "n * n * sizeof(*A)"},
    "cublasDtpttr": {"AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "A": "n * n * sizeof(*A)"},
    "cublasCtpttr": {"AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "A": "n * n * sizeof(*A)"},
    "cublasZtpttr": {"AP": "(n * (n + 1)) / 2 * sizeof(*AP)", "A": "n * n * sizeof(*A)"},
    "cublasStrttp": {"A": "n * n * sizeof(*A)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)"},
    "cublasDtrttp": {"A": "n * n * sizeof(*A)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)"},
    "cublasCtrttp": {"A": "n * n * sizeof(*A)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)"},
    "cublasZtrttp": {"A": "n * n * sizeof(*A)", "AP": "(n * (n + 1)) / 2 * sizeof(*AP)"},
    # cuda.h
    "cuDriverGetVersion": {"driverVersion": "sizeof(*driverVersion)"},
    "cuDeviceGet": {"device": "sizeof(*device)"},
    "cuDeviceGetCount": {"count": "sizeof(*count)"},
    "cuDeviceGetName": {"name": "len"},
    "cuDeviceGetUuid": {"uuid": "sizeof(*uuid)"},
    "cuDeviceGetUuid_v2": {"uuid": "sizeof(*uuid)"},
    "cuDeviceGetLuid": {"luid": "sizeof(*luid)", "deviceNodeMask": "sizeof(*deviceNodeMask)"},
    "cuDeviceTotalMem_v2": {"bytes": "sizeof(*bytes)"},
    "cuDeviceGetTexture1DLinearMaxWidth": {"maxWidthInElements": "sizeof(*maxWidthInElements)"},
    "cuDeviceGetAttribute": {"pi": "sizeof(*pi)"},
    "cuDeviceGetNvSciSyncAttributes": {"nvSciSyncAttrList": "sizeof(*nvSciSyncAttrList)"},
    "cuDeviceSetMemPool": {"pool": "sizeof(*pool)"},
    "cuDeviceGetMemPool": {"pool": "sizeof(*pool)"},
    "cuDeviceGetDefaultMemPool": {"pool_out": "sizeof(*pool_out)"},
    "cuDeviceGetProperties": {"prop": "sizeof(*prop)"},
    "cuDeviceComputeCapability": {"major": "sizeof(*major)", "minor": "sizeof(*minor)"},
    "cuDevicePrimaryCtxRetain": {"pctx": "sizeof(*pctx)"},
    "cuDevicePrimaryCtxGetState": {"flags": "sizeof(*flags)", "active": "sizeof(*active)"},
    "cuDeviceGetExecAffinitySupport": {"pi": "sizeof(*pi)"},
    "cuCtxCreate_v2": {"pctx": "sizeof(*pctx)"},
    "cuCtxCreate_v3": {"pctx": "sizeof(*pctx)", "paramsArray": "numParams * sizeof(*paramsArray)"},
    "cuCtxPopCurrent_v2": {"pctx": "sizeof(*pctx)"},
    "cuCtxGetCurrent": {"pctx": "sizeof(*pctx)"},
    "cuCtxGetDevice": {"device": "sizeof(*device)"},
    "cuCtxGetFlags": {"flags": "sizeof(*flags)"},
    "cuCtxGetLimit": {"pvalue": "sizeof(*pvalue)"},
    "cuCtxGetCacheConfig": {"pconfig": "sizeof(*pconfig)"},
    "cuCtxGetSharedMemConfig": {"pConfig": "sizeof(*pConfig)"},
    "cuCtxGetApiVersion": {"version": "sizeof(*version)"},
    "cuCtxGetStreamPriorityRange": {"leastPriority": "sizeof(*leastPriority)", "greatestPriority": "sizeof(*greatestPriority)"},
    "cuCtxGetExecAffinity": {"pExecAffinity": "sizeof(*pExecAffinity)"},
    "cuCtxAttach": {"pctx": "sizeof(*pctx)"},
    "cuModuleLoad": {"module": "sizeof(*module)", "fname": "strlen(fname) + 1"},
    "cuModuleLoadData": {"module": "sizeof(*module)", "image": "sizeof(*image)"},
    "cuModuleLoadDataEx": {"module": "sizeof(*module)", "image": "sizeof(*image)", "options": "numOptions * sizeof(*options)", "optionValues": "numOptions * sizeof(*optionValues)"},
    "cuModuleLoadFatBinary": {"module": "sizeof(*module)", "fatCubin": "sizeof(*fatCubin)"},
    "cuModuleGetFunction": {"hfunc": "sizeof(*hfunc)", "name": "strlen(name) + 1"},
    "cuModuleGetTexRef": {"pTexRef": "sizeof(*pTexRef)", "name": "strlen(name) + 1"},
    "cuModuleGetSurfRef": {"pSurfRef": "sizeof(*pSurfRef)", "name": "strlen(name) + 1"},
    "cuLinkCreate_v2": {"options": "numOptions * sizeof(*options)", "optionValues": "numOptions * sizeof(*optionValues)", "stateOut": "sizeof(*stateOut)"},
    "cuLinkAddData_v2": {"data": "size", "name": "strlen(name) + 1", "options": "numOptions * sizeof(*options)", "optionValues": "numOptions * sizeof(*optionValues)"},
    "cuLinkAddFile_v2": {"path": "strlen(path) + 1", "options": "numOptions * sizeof(*options)", "optionValues": "numOptions * sizeof(*optionValues)"},
    "cuLinkComplete": {"cubinOut": "sizeof(*cubinOut)", "sizeOut": "sizeof(*sizeOut)"},
    "cuMemGetInfo_v2": {"free": "sizeof(*free)", "total": "sizeof(*total)"},
    "cuMemHostGetFlags": {"pFlags": "sizeof(*pFlags)", "p": "sizeof(*p)"},
    "cuDeviceGetByPCIBusId": {"dev": "sizeof(*dev)", "pciBusId": "strlen(pciBusId) + 1"},
    "cuDeviceGetPCIBusId": {"pciBusId": "len"},
    "cuIpcGetEventHandle": {"pHandle": "sizeof(*pHandle)"},
    "cuIpcOpenEventHandle": {"phEvent": "sizeof(*phEvent)"},
    "cuIpcGetMemHandle": {"pHandle": "sizeof(*pHandle)"},
    "cuMemHostRegister_v2": {"p": "bytesize"},
    "cuMemHostUnregister": {"p": "sizeof(*p)"},
    "cuMemcpyHtoD_v2": {"srcHost": "ByteCount"},
    "cuMemcpyDtoH_v2": {"dstHost": "ByteCount"},
    "cuMemcpyHtoA_v2": {"srcHost": "ByteCount"},
    "cuMemcpyAtoH_v2": {"dstHost": "ByteCount"},
    "cuMemcpy2D_v2": {"pCopy": "sizeof(*pCopy)"},
    "cuMemcpy2DUnaligned_v2": {"pCopy": "sizeof(*pCopy)"},
    "cuMemcpy3D_v2": {"pCopy": "sizeof(*pCopy)"},
    "cuMemcpy3DPeer": {"pCopy": "sizeof(*pCopy)"},
    "cuMemcpyHtoDAsync_v2": {"srcHost": "ByteCount"},
    "cuMemcpyDtoHAsync_v2": {"dstHost": "ByteCount"},
    "cuMemcpyHtoAAsync_v2": {"srcHost": "ByteCount"},
    "cuMemcpyAtoHAsync_v2": {"dstHost": "ByteCount"},
    "cuMemcpy2DAsync_v2": {"pCopy": "sizeof(*pCopy)"},
    "cuMemcpy3DAsync_v2": {"pCopy": "sizeof(*pCopy)"},
    "cuMemcpy3DPeerAsync": {"pCopy": "sizeof(*pCopy)"},
    "cuArrayCreate_v2": {"pHandle": "sizeof(*pHandle)", "pAllocateArray": "sizeof(*pAllocateArray)"},
    "cuArrayGetDescriptor_v2": {"pArrayDescriptor": "sizeof(*pArrayDescriptor)"},
    "cuArrayGetSparseProperties": {"sparseProperties": "sizeof(*sparseProperties)"},
    "cuMipmappedArrayGetSparseProperties": {"sparseProperties": "sizeof(*sparseProperties)"},
    "cuArrayGetPlane": {"pPlaneArray": "sizeof(*pPlaneArray)"},
    "cuArray3DCreate_v2": {"pHandle": "sizeof(*pHandle)", "pAllocateArray": "sizeof(*pAllocateArray)"},
    "cuArray3DGetDescriptor_v2": {"pArrayDescriptor": "sizeof(*pArrayDescriptor)"},
    "cuMipmappedArrayCreate": {"pHandle": "sizeof(*pHandle)", "pMipmappedArrayDesc": "sizeof(*pMipmappedArrayDesc)"},
    "cuMipmappedArrayGetLevel": {"pLevelArray": "sizeof(*pLevelArray)"},
    "cuMemMapArrayAsync": {"mapInfoList": "count * sizeof(*mapInfoList)"},
    "cuMemSetAccess": {"desc": "count * sizeof(*desc)"},
    "cuMemGetAccess": {"flags": "sizeof(*flags)", "location": "sizeof(*location)"},
    "cuMemExportToShareableHandle": {"shareableHandle": "sizeof(*shareableHandle)"},
    "cuMemImportFromShareableHandle": {"handle": "sizeof(*handle)", "osHandle": "sizeof(*osHandle)"},
    "cuMemGetAllocationGranularity": {"granularity": "sizeof(*granularity)", "prop": "sizeof(*prop)"},
    "cuMemGetAllocationPropertiesFromHandle": {"prop": "sizeof(*prop)"},
    "cuMemRetainAllocationHandle": {"handle": "sizeof(*handle)", "addr": "sizeof(*addr)"},
    "cuMemPoolSetAttribute": {"value": "sizeof(*value)"},
    "cuMemPoolGetAttribute": {"value": "sizeof(*value)"},
    "cuMemPoolSetAccess": {"map": "count * sizeof(*map)"},
    "cuMemPoolGetAccess": {"flags": "sizeof(*flags)", "location": "sizeof(*location)"},
    "cuMemPoolCreate": {"pool": "sizeof(*pool)", "poolProps": "sizeof(*poolProps)"},
    "cuMemPoolExportToShareableHandle": {"handle_out": "sizeof(*handle_out)"},
    "cuMemPoolImportFromShareableHandle": {"pool_out": "sizeof(*pool_out)", "handle": "sizeof(*handle)"},
    "cuMemPoolExportPointer": {"shareData_out": "sizeof(*shareData_out)"},
    "cuMemRangeGetAttribute": {"data": "dataSize"},
    "cuPointerSetAttribute": {"value": "sizeof(*value)"},
    "cuPointerGetAttributes": {"attributes": "numAttributes * sizeof(*attributes)", "data": "numAttributes * sizeof(*data)"},
    "cuStreamCreate": {"phStream": "sizeof(*phStream)"},
    "cuStreamCreateWithPriority": {"phStream": "sizeof(*phStream)"},
    "cuStreamGetPriority": {"priority": "sizeof(*priority)"},
    "cuStreamGetFlags": {"flags": "sizeof(*flags)"},
    "cuStreamGetCtx": {"pctx": "sizeof(*pctx)"},
    "cuStreamAddCallback": {"userData": "sizeof(*userData)"},
    "cuThreadExchangeStreamCaptureMode": {"mode": "sizeof(*mode)"},
    "cuStreamEndCapture": {"phGraph": "sizeof(*phGraph)"},
    "cuStreamIsCapturing": {"captureStatus": "sizeof(*captureStatus)"},
    "cuStreamGetCaptureInfo": {"captureStatus_out": "sizeof(*captureStatus_out)", "id_out": "sizeof(*id_out)"},
    "cuStreamGetCaptureInfo_v2": {
        "captureStatus_out": "sizeof(*captureStatus_out)",
        "id_out": "sizeof(*id_out)",
        "graph_out": "sizeof(*graph_out)",
        "dependencies_out": "numDependencies_out * sizeof(*dependencies_out)",
        "numDependencies_out": "sizeof(*numDependencies_out)",
    },
    "cuStreamUpdateCaptureDependencies": {"dependencies": "numDependencies * sizeof(*dependencies)"},
    "cuStreamGetAttribute": {"value_out": "sizeof(*value_out)"},
    "cuStreamSetAttribute": {"value": "sizeof(*value)"},
    "cuEventCreate": {"phEvent": "sizeof(*phEvent)"},
    "cuEventElapsedTime": {"pMilliseconds": "sizeof(*pMilliseconds)"},
    "cuExternalMemoryGetMappedMipmappedArray": {"mipmap": "sizeof(*mipmap)", "mipmapDesc": "sizeof(*mipmapDesc)"},
    "cuImportExternalSemaphore": {"extSem_out": "sizeof(*extSem_out)", "semHandleDesc": "sizeof(*semHandleDesc)"},
    "cuSignalExternalSemaphoresAsync": {"extSemArray": "numExtSems * sizeof(*extSemArray)", "paramsArray": "numExtSems * sizeof(*paramsArray)"},
    "cuWaitExternalSemaphoresAsync": {"extSemArray": "numExtSems * sizeof(*extSemArray)", "paramsArray": "numExtSems * sizeof(*paramsArray)"},
    "cuStreamBatchMemOp": {"paramArray": "count * sizeof(*paramArray)"},
    "cuFuncGetAttribute": {"pi": "sizeof(*pi)"},
    "cuFuncGetModule": {"hmod": "sizeof(*hmod)"},
    "cuLaunchKernel": {"kernelParams": "sizeof(*kernelParams)", "extra": "sizeof(*extra)"},
    "cuLaunchCooperativeKernel": {"kernelParams": "sizeof(*kernelParams)"},
    "cuLaunchCooperativeKernelMultiDevice": {"launchParamsList": "numDevices * sizeof(*launchParamsList)"},
    "cuLaunchHostFunc": {"userData": "sizeof(*userData)"},
    "cuParamSetv": {"ptr": "numbytes"},
    "cuGraphCreate": {"phGraph": "sizeof(*phGraph)"},
    "cuGraphAddKernelNode": {"phGraphNode": "sizeof(*phGraphNode)", "dependencies": "numDependencies * sizeof(*dependencies)", "nodeParams": "sizeof(*nodeParams)"},
    "cuGraphKernelNodeGetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cuGraphKernelNodeSetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cuGraphAddMemcpyNode": {"phGraphNode": "sizeof(*phGraphNode)", "dependencies": "numDependencies * sizeof(*dependencies)", "copyParams": "sizeof(*copyParams)"},
    "cuGraphMemcpyNodeGetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cuGraphMemcpyNodeSetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cuGraphAddMemsetNode": {"phGraphNode": "sizeof(*phGraphNode)", "dependencies": "numDependencies * sizeof(*dependencies)", "memsetParams": "sizeof(*memsetParams)"},
    "cuGraphMemsetNodeGetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cuGraphMemsetNodeSetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cuGraphAddHostNode": {"phGraphNode": "sizeof(*phGraphNode)", "dependencies": "numDependencies * sizeof(*dependencies)", "nodeParams": "sizeof(*nodeParams)"},
    "cuGraphHostNodeGetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cuGraphHostNodeSetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cuGraphAddChildGraphNode": {"phGraphNode": "sizeof(*phGraphNode)", "dependencies": "numDependencies * sizeof(*dependencies)"},
    "cuGraphChildGraphNodeGetGraph": {"phGraph": "sizeof(*phGraph)"},
    "cuGraphAddEmptyNode": {"phGraphNode": "sizeof(*phGraphNode)", "dependencies": "numDependencies * sizeof(*dependencies)"},
    "cuGraphAddEventRecordNode": {"phGraphNode": "sizeof(*phGraphNode)", "dependencies": "numDependencies * sizeof(*dependencies)"},
    "cuGraphEventRecordNodeGetEvent": {"event_out": "sizeof(*event_out)"},
    "cuGraphAddEventWaitNode": {"phGraphNode": "sizeof(*phGraphNode)", "dependencies": "numDependencies * sizeof(*dependencies)"},
    "cuGraphEventWaitNodeGetEvent": {"event_out": "sizeof(*event_out)"},
    "cuGraphAddExternalSemaphoresSignalNode": {"phGraphNode": "sizeof(*phGraphNode)", "dependencies": "numDependencies * sizeof(*dependencies)", "nodeParams": "sizeof(*nodeParams)"},
    "cuGraphExternalSemaphoresSignalNodeGetParams": {"params_out": "sizeof(*params_out)"},
    "cuGraphExternalSemaphoresSignalNodeSetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cuGraphAddExternalSemaphoresWaitNode": {"phGraphNode": "sizeof(*phGraphNode)", "dependencies": "numDependencies * sizeof(*dependencies)", "nodeParams": "sizeof(*nodeParams)"},
    "cuGraphExternalSemaphoresWaitNodeGetParams": {"params_out": "sizeof(*params_out)"},
    "cuGraphExternalSemaphoresWaitNodeSetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cuGraphAddMemAllocNode": {"phGraphNode": "sizeof(*phGraphNode)", "dependencies": "numDependencies * sizeof(*dependencies)", "nodeParams": "sizeof(*nodeParams)"},
    "cuGraphMemAllocNodeGetParams": {"params_out": "sizeof(*params_out)"},
    "cuGraphAddMemFreeNode": {"phGraphNode": "sizeof(*phGraphNode)", "dependencies": "numDependencies * sizeof(*dependencies)"},
    "cuDeviceGetGraphMemAttribute": {"value": "sizeof(*value)"},
    "cuDeviceSetGraphMemAttribute": {"value": "sizeof(*value)"},
    "cuGraphClone": {"phGraphClone": "sizeof(*phGraphClone)"},
    "cuGraphNodeFindInClone": {"phNode": "sizeof(*phNode)"},
    "cuGraphNodeGetType": {"type": "sizeof(*type)"},
    "cuGraphGetNodes": {"nodes": "numNodes * sizeof(*nodes)", "numNodes": "sizeof(*numNodes)"},
    "cuGraphGetRootNodes": {"rootNodes": "numRootNodes * sizeof(*rootNodes)", "numRootNodes": "sizeof(*numRootNodes)"},
    "cuGraphGetEdges": {"from": "numEdges * sizeof(*from)", "to": "numEdges * sizeof(*to)", "numEdges": "sizeof(*numEdges)"},
    "cuGraphNodeGetDependencies": {"dependencies": "numDependencies * sizeof(*dependencies)", "numDependencies": "sizeof(*numDependencies)"},
    "cuGraphNodeGetDependentNodes": {"dependentNodes": "numDependentNodes * sizeof(*dependentNodes)", "numDependentNodes": "sizeof(*numDependentNodes)"},
    "cuGraphAddDependencies": {"from": "numDependencies * sizeof(*from)", "to": "numDependencies * sizeof(*to)"},
    "cuGraphRemoveDependencies": {"from": "numDependencies * sizeof(*from)", "to": "numDependencies * sizeof(*to)"},
    "cuGraphInstantiate_v2": {"phGraphExec": "sizeof(*phGraphExec)", "phErrorNode": "sizeof(*phErrorNode)", "logBuffer": "bufferSize"},
    "cuGraphInstantiateWithFlags": {"phGraphExec": "sizeof(*phGraphExec)"},
    "cuGraphExecKernelNodeSetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cuGraphExecMemcpyNodeSetParams": {"copyParams": "sizeof(*copyParams)"},
    "cuGraphExecMemsetNodeSetParams": {"memsetParams": "sizeof(*memsetParams)"},
    "cuGraphExecHostNodeSetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cuGraphExecExternalSemaphoresSignalNodeSetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cuGraphExecExternalSemaphoresWaitNodeSetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cuGraphExecUpdate": {"hErrorNode_out": "sizeof(*hErrorNode_out)", "updateResult_out": "sizeof(*updateResult_out)"},
    "cuGraphKernelNodeGetAttribute": {"value_out": "sizeof(*value_out)"},
    "cuGraphKernelNodeSetAttribute": {"value": "sizeof(*value)"},
    "cuGraphDebugDotPrint": {"path": "strlen(path) + 1"},
    "cuUserObjectCreate": {"object_out": "sizeof(*object_out)"},
    "cuOccupancyMaxActiveBlocksPerMultiprocessor": {"numBlocks": "sizeof(*numBlocks)"},
    "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags": {"numBlocks": "sizeof(*numBlocks)"},
    "cuOccupancyMaxPotentialBlockSize": {"minGridSize": "sizeof(*minGridSize)", "blockSize": "sizeof(*blockSize)"},
    "cuOccupancyMaxPotentialBlockSizeWithFlags": {"minGridSize": "sizeof(*minGridSize)", "blockSize": "sizeof(*blockSize)"},
    "cuOccupancyAvailableDynamicSMemPerBlock": {"dynamicSmemSize": "sizeof(*dynamicSmemSize)"},
    "cuTexRefSetAddress_v2": {"ByteOffset": "sizeof(*ByteOffset)"},
    "cuTexRefSetAddress2D_v3": {"desc": "sizeof(*desc)"},
    "cuTexRefSetBorderColor": {"pBorderColor": "sizeof(*pBorderColor)"},
    "cuTexRefGetArray": {"phArray": "sizeof(*phArray)"},
    "cuTexRefGetMipmappedArray": {"phMipmappedArray": "sizeof(*phMipmappedArray)"},
    "cuTexRefGetAddressMode": {"pam": "sizeof(*pam)"},
    "cuTexRefGetFilterMode": {"pfm": "sizeof(*pfm)"},
    "cuTexRefGetFormat": {"pFormat": "sizeof(*pFormat)", "pNumChannels": "sizeof(*pNumChannels)"},
    "cuTexRefGetMipmapFilterMode": {"pfm": "sizeof(*pfm)"},
    "cuTexRefGetMipmapLevelBias": {"pbias": "sizeof(*pbias)"},
    "cuTexRefGetMipmapLevelClamp": {"pminMipmapLevelClamp": "sizeof(*pminMipmapLevelClamp)", "pmaxMipmapLevelClamp": "sizeof(*pmaxMipmapLevelClamp)"},
    "cuTexRefGetMaxAnisotropy": {"pmaxAniso": "sizeof(*pmaxAniso)"},
    "cuTexRefGetBorderColor": {"pBorderColor": "sizeof(*pBorderColor)"},
    "cuTexRefGetFlags": {"pFlags": "sizeof(*pFlags)"},
    "cuTexRefCreate": {"pTexRef": "sizeof(*pTexRef)"},
    "cuSurfRefGetArray": {"phArray": "sizeof(*phArray)"},
    "cuTexObjectCreate": {"pTexObject": "sizeof(*pTexObject)", "pResDesc": "sizeof(*pResDesc)", "pTexDesc": "sizeof(*pTexDesc)", "pResViewDesc": "sizeof(*pResViewDesc)"},
    "cuTexObjectGetResourceDesc": {"pResDesc": "sizeof(*pResDesc)"},
    "cuTexObjectGetTextureDesc": {"pTexDesc": "sizeof(*pTexDesc)"},
    "cuTexObjectGetResourceViewDesc": {"pResViewDesc": "sizeof(*pResViewDesc)"},
    "cuSurfObjectCreate": {"pSurfObject": "sizeof(*pSurfObject)", "pResDesc": "sizeof(*pResDesc)"},
    "cuSurfObjectGetResourceDesc": {"pResDesc": "sizeof(*pResDesc)"},
    "cuDeviceCanAccessPeer": {"canAccessPeer": "sizeof(*canAccessPeer)"},
    "cuDeviceGetP2PAttribute": {"value": "sizeof(*value)"},
    "cuGraphicsSubResourceGetMappedArray": {"pArray": "sizeof(*pArray)"},
    "cuGraphicsResourceGetMappedMipmappedArray": {"pMipmappedArray": "sizeof(*pMipmappedArray)"},
    "cuGraphicsMapResources": {"resources": "count * sizeof(*resources)"},
    "cuGraphicsUnmapResources": {"resources": "count * sizeof(*resources)"},
    "cuGetExportTable": {"ppExportTable": "sizeof(*ppExportTable)", "pExportTableId": "sizeof(*pExportTableId)"},
    # cuda runtime api
    "cudaDeviceGetLimit": {"pValue": "sizeof(*pValue)"},
    "cudaDeviceGetTexture1DLinearMaxWidth": {"maxWidthInElements": "sizeof(*maxWidthInElements)", "fmtDesc": "sizeof(*fmtDesc)"},
    "cudaDeviceGetCacheConfig": {"pCacheConfig": "sizeof(*pCacheConfig)"},
    "cudaDeviceGetStreamPriorityRange": {"leastPriority": "sizeof(*leastPriority)", "greatestPriority": "sizeof(*greatestPriority)"},
    "cudaDeviceGetSharedMemConfig": {"pConfig": "sizeof(*pConfig)"},
    "cudaDeviceGetByPCIBusId": {"device": "sizeof(*device)", "pciBusId": "strlen(pciBusId) + 1"},
    "cudaDeviceGetPCIBusId": {"pciBusId": "len"},
    "cudaIpcGetEventHandle": {"handle": "sizeof(*handle)"},
    "cudaIpcOpenEventHandle": {"event": "sizeof(*event)"},
    "cudaIpcGetMemHandle": {"handle": "sizeof(*handle)", "devPtr": "-1"},
    "cudaIpcCloseMemHandle": {"devPtr": "-1"},
    "cudaThreadGetLimit": {"pValue": "sizeof(*pValue)"},
    "cudaThreadGetCacheConfig": {"pCacheConfig": "sizeof(*pCacheConfig)"},
    "cudaGetDeviceCount": {"count": "sizeof(*count)"},
    "cudaGetDeviceProperties": {"prop": "sizeof(*prop)"},
    "cudaDeviceGetAttribute": {"value": "sizeof(*value)"},
    "cudaDeviceGetDefaultMemPool": {"memPool": "sizeof(*memPool)"},
    "cudaDeviceSetMemPool": {"memPool": "sizeof(*memPool)"},
    "cudaDeviceGetMemPool": {"memPool": "sizeof(*memPool)"},
    "cudaDeviceGetNvSciSyncAttributes": {"nvSciSyncAttrList": "sizeof(*nvSciSyncAttrList)"},
    "cudaDeviceGetP2PAttribute": {"value": "sizeof(*value)"},
    "cudaChooseDevice": {"device": "sizeof(*device)", "prop": "sizeof(*prop)"},
    "cudaSetDevice": {"device": "sizeof(*device)"},
    "cudaGetDevice": {"device": "sizeof(*device)"},
    "cudaSetValidDevices": {"device_arr": "len * sizeof(*device_arr)"},
    "cudaGetDeviceFlags": {"flags": "sizeof(*flags)"},
    "cudaStreamCreate": {"pStream": "sizeof(*pStream)"},
    "cudaStreamCreateWithFlags": {"pStream": "sizeof(*pStream)"},
    "cudaStreamCreateWithPriority": {"pStream": "sizeof(*pStream)"},
    "cudaStreamGetPriority": {"priority": "sizeof(*priority)"},
    "cudaStreamGetFlags": {"flags": "sizeof(*flags)"},
    "cudaStreamCopyAttributes": {"dst": "sizeof(*dst)", "src": "sizeof(*src)"},
    "cudaStreamGetAttribute": {"value_out": "sizeof(*value_out)"},
    "cudaStreamSetAttribute": {"value": "sizeof(*value)"},
    "cudaStreamDestroy": {"stream": "sizeof(*stream)"},
    "cudaStreamWaitEvent": {"stream": "sizeof(*stream)", "event": "sizeof(*event)"},
    "cudaStreamAddCallback": {"stream": "sizeof(*stream)", "callback": "sizeof(*callback)", "userData": "sizeof(*userData)"},
    "cudaStreamSynchronize": {"stream": "sizeof(*stream)"},
    "cudaStreamQuery": {"stream": "sizeof(*stream)"},
    "cudaStreamAttachMemAsync": {"stream": "sizeof(*stream)", "devPtr": "length"},
    "cudaStreamBeginCapture": {"stream": "sizeof(*stream)"},
    "cudaThreadExchangeStreamCaptureMode": {"mode": "sizeof(*mode)"},
    "cudaStreamEndCapture": {"stream": "sizeof(*stream)", "pGraph": "sizeof(*pGraph)"},
    "cudaStreamIsCapturing": {"stream": "sizeof(*stream)", "pCaptureStatus": "sizeof(*pCaptureStatus)"},
    "cudaStreamGetCaptureInfo": {"stream": "sizeof(*stream)", "pCaptureStatus": "sizeof(*pCaptureStatus)", "pId": "sizeof(*pId)"},
    "cudaStreamGetCaptureInfo_v2": {
        "stream": "sizeof(*stream)",
        "captureStatus_out": "sizeof(*captureStatus_out)",
        "id_out": "sizeof(*id_out)",
        "graph_out": "sizeof(*graph_out)",
        "dependencies_out": "numDependencies_out * sizeof(*dependencies_out)",
        "numDependencies_out": "sizeof(*numDependencies_out)",
    },
    "cudaStreamUpdateCaptureDependencies": {"stream": "sizeof(*stream)", "dependencies": "numDependencies * sizeof(*dependencies)"},
    "cudaEventCreate": {"event": "sizeof(*event)"},
    "cudaEventCreateWithFlags": {"event": "sizeof(*event)"},
    "cudaEventRecord": {"event": "sizeof(*event)", "stream": "sizeof(*stream)"},
    "cudaEventRecordWithFlags": {"event": "sizeof(*event)", "stream": "sizeof(*stream)"},
    "cudaEventQuery": {"event": "sizeof(*event)"},
    "cudaEventSynchronize": {"event": "sizeof(*event)"},
    "cudaEventDestroy": {"event": "sizeof(*event)"},
    "cudaEventElapsedTime": {"ms": "sizeof(*ms)", "start": "sizeof(*start)", "end": "sizeof(*end)"},
    "cudaImportExternalMemory": {"extMem_out": "sizeof(*extMem_out)", "memHandleDesc": "sizeof(*memHandleDesc)"},
    "cudaExternalMemoryGetMappedBuffer": {"devPtr": "sizeof(*devPtr)", "bufferDesc": "sizeof(*bufferDesc)"},
    "cudaExternalMemoryGetMappedMipmappedArray": {"mipmap": "sizeof(*mipmap)", "mipmapDesc": "sizeof(*mipmapDesc)"},
    "cudaDestroyExternalMemory": {"extMem": "sizeof(*extMem)"},
    "cudaImportExternalSemaphore": {"extSem_out": "sizeof(*extSem_out)", "semHandleDesc": "sizeof(*semHandleDesc)"},
    "cudaSignalExternalSemaphoresAsync_v2": {"extSemArray": "numExtSems * sizeof(*extSemArray)", "paramsArray": "numExtSems * sizeof(*paramsArray)", "stream": "sizeof(*stream)"},
    "cudaWaitExternalSemaphoresAsync_v2": {"extSemArray": "numExtSems * sizeof(*extSemArray)", "paramsArray": "numExtSems * sizeof(*paramsArray)", "stream": "sizeof(*stream)"},
    "cudaDestroyExternalSemaphore": {"extSem": "sizeof(*extSem)"},
    "cudaLaunchCooperativeKernel": {"func": "sizeof(*func)", "args": "sizeof(*args)", "stream": "sizeof(*stream)"},
    "cudaLaunchCooperativeKernelMultiDevice": {"launchParamsList": "numDevices * sizeof(*launchParamsList)"},
    "cudaFuncSetCacheConfig": {"func": "sizeof(*func)"},
    "cudaFuncSetSharedMemConfig": {"func": "sizeof(*func)"},
    "cudaFuncGetAttributes": {"attr": "sizeof(*attr)", "func": "sizeof(*func)"},
    "cudaFuncSetAttribute": {"func": "sizeof(*func)"},
    "cudaSetDoubleForDevice": {"d": "sizeof(*d)"},
    "cudaSetDoubleForHost": {"d": "sizeof(*d)"},
    "cudaLaunchHostFunc": {"stream": "sizeof(*stream)", "fn": "sizeof(*fn)", "userData": "sizeof(*userData)"},
    "cudaOccupancyMaxActiveBlocksPerMultiprocessor": {"numBlocks": "sizeof(*numBlocks)", "func": "sizeof(*func)"},
    "cudaOccupancyAvailableDynamicSMemPerBlock": {"dynamicSmemSize": "sizeof(*dynamicSmemSize)", "func": "sizeof(*func)"},
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags": {"numBlocks": "sizeof(*numBlocks)", "func": "sizeof(*func)"},
    "cudaMallocArray": {"array": "sizeof(*array)", "desc": "sizeof(*desc)"},
    "cudaFreeArray": {"array": "sizeof(*array)"},
    "cudaFreeMipmappedArray": {"mipmappedArray": "sizeof(*mipmappedArray)"},
    "cudaHostRegister": {"ptr": "size"},
    "cudaHostGetDevicePointer": {"pDevice": "sizeof(*pDevice)", "pHost": "sizeof(*pHost)"},
    "cudaHostGetFlags": {"pFlags": "sizeof(*pFlags)", "pHost": "sizeof(*pHost)"},
    "cudaMalloc3DArray": {"array": "sizeof(*array)", "desc": "sizeof(*desc)"},
    "cudaMallocMipmappedArray": {"mipmappedArray": "sizeof(*mipmappedArray)", "desc": "sizeof(*desc)"},
    "cudaGetMipmappedArrayLevel": {"levelArray": "sizeof(*levelArray)", "mipmappedArray": "sizeof(*mipmappedArray)"},
    "cudaMemcpy3D": {"p": "sizeof(*p)"},
    "cudaMemcpy3DPeer": {"p": "sizeof(*p)"},
    "cudaMemcpy3DAsync": {"p": "sizeof(*p)", "stream": "sizeof(*stream)"},
    "cudaMemcpy3DPeerAsync": {"p": "sizeof(*p)", "stream": "sizeof(*stream)"},
    "cudaMemGetInfo": {"free": "sizeof(*free)", "total": "sizeof(*total)"},
    "cudaArrayGetInfo": {"desc": "sizeof(*desc)", "extent": "sizeof(*extent)", "flags": "sizeof(*flags)", "array": "sizeof(*array)"},
    "cudaArrayGetPlane": {"pPlaneArray": "sizeof(*pPlaneArray)", "hArray": "sizeof(*hArray)"},
    "cudaArrayGetSparseProperties": {"sparseProperties": "sizeof(*sparseProperties)", "array": "sizeof(*array)"},
    "cudaMipmappedArrayGetSparseProperties": {"sparseProperties": "sizeof(*sparseProperties)", "mipmap": "sizeof(*mipmap)"},
    "cudaMemcpy": {"dst": "count", "src": "count"},
    "cudaMemcpyPeer": {"dst": "count", "src": "count"},
    "cudaMemcpy2D": {"dst": "dpitch * height", "src": "spitch * height"},
    "cudaMemcpy2DToArray": {"dst": "sizeof(*dst)", "src": "spitch * height"},
    "cudaMemcpy2DFromArray": {"dst": "dpitch * height", "src": "sizeof(*src)"},
    "cudaMemcpy2DArrayToArray": {"dst": "sizeof(*dst)", "src": "sizeof(*src)"},
    "cudaMemcpyAsync": {"dst": "count", "src": "count", "stream": "sizeof(*stream)"},
    "cudaMemcpyPeerAsync": {"dst": "count", "src": "count", "stream": "sizeof(*stream)"},
    "cudaMemcpy2DAsync": {"dst": "dpitch * height", "src": "spitch * height", "stream": "sizeof(*stream)"},
    "cudaMemcpy2DToArrayAsync": {"dst": "sizeof(*dst)", "src": "spitch * height", "stream": "sizeof(*stream)"},
    "cudaMemcpy2DFromArrayAsync": {"dst": "dpitch * height", "src": "sizeof(*src)", "stream": "sizeof(*stream)"},
    "cudaMemcpyToSymbol": {"symbol": "-1", "src": "count", "stream": "sizeof(*stream)"},
    "cudaMemcpyToSymbolAsync": {"symbol": "-1", "src": "count", "stream": "sizeof(*stream)"},
    "cudaMemcpyFromSymbol": {"symbol": "-1", "dst": "count", "stream": "sizeof(*stream)"},
    "cudaMemcpyFromSymbolAsync": {"dst": "count", "symbol": "-1", "stream": "sizeof(*stream)"},
    "cudaMemset": {"devPtr": "count"},
    "cudaMemsetAsync": {"devPtr": "count"},
    "cudaMemset2D": {"devPtr": "pitch * height"},
    "cudaMemset3D": {"pitchedDevPtr": "sizeof(*pitchedDevPtr)"},
    "cudaMemset2DAsync": {"devPtr": "pitch * height", "stream": "sizeof(*stream)"},
    "cudaMemset3DAsync": {"pitchedDevPtr": "sizeof(*pitchedDevPtr)", "stream": "sizeof(*stream)"},
    "cudaGetSymbolSize": {"size": "sizeof(*size)", "symbol": "-1"},
    "cudaMemPrefetchAsync": {"devPtr": "count", "stream": "sizeof(*stream)"},
    "cudaMemAdvise": {"devPtr": "count"},
    "cudaMemRangeGetAttribute": {"data": "dataSize", "devPtr": "count"},
    "cudaMemRangeGetAttributes": {"data": "numAttributes * sizeof(*data)", "dataSizes": "numAttributes * sizeof(*dataSizes)", "attributes": "numAttributes * sizeof(*attributes)", "devPtr": "count"},
    "cudaMemcpyToArray": {"dst": "sizeof(*dst)", "src": "count"},
    "cudaMemcpyFromArray": {"dst": "count", "src": "sizeof(*src)"},
    "cudaMemcpyArrayToArray": {"dst": "sizeof(*dst)", "src": "sizeof(*src)"},
    "cudaMemcpyToArrayAsync": {"dst": "sizeof(*dst)", "src": "count", "stream": "sizeof(*stream)"},
    "cudaMemcpyFromArrayAsync": {"dst": "count", "src": "sizeof(*src)", "stream": "sizeof(*stream)"},
    "cudaMallocAsync": {"devPtr": "size", "hStream": "sizeof(*hStream)"},
    "cudaFreeAsync": {"hStream": "sizeof(*hStream)", "devPtr": "-1"},
    "cudaMemPoolTrimTo": {"memPool": "sizeof(*memPool)"},
    "cudaMemPoolSetAttribute": {"memPool": "sizeof(*memPool)", "value": "sizeof(*value)"},
    "cudaMemPoolGetAttribute": {"memPool": "sizeof(*memPool)", "value": "sizeof(*value)"},
    "cudaMemPoolSetAccess": {"memPool": "sizeof(*memPool)", "descList": "count * sizeof(*descList)"},
    "cudaMemPoolGetAccess": {"flags": "sizeof(*flags)", "memPool": "sizeof(*memPool)", "location": "sizeof(*location)"},
    "cudaMemPoolCreate": {"memPool": "sizeof(*memPool)", "poolProps": "sizeof(*poolProps)"},
    "cudaMemPoolDestroy": {"memPool": "sizeof(*memPool)"},
    "cudaMallocFromPoolAsync": {"ptr": "size", "memPool": "sizeof(*memPool)", "stream": "sizeof(*stream)"},
    "cudaMemPoolExportToShareableHandle": {"shareableHandle": "sizeof(*shareableHandle)", "memPool": "sizeof(*memPool)"},
    "cudaMemPoolImportFromShareableHandle": {"memPool": "sizeof(*memPool)", "shareableHandle": "sizeof(*shareableHandle)"},
    "cudaMemPoolExportPointer": {"exportData": "sizeof(*exportData)"},
    "cudaMemPoolImportPointer": {"ptr": "sizeof(*ptr)", "memPool": "sizeof(*memPool)", "exportData": "sizeof(*exportData)"},
    "cudaPointerGetAttributes": {"attributes": "sizeof(*attributes)"},
    "cudaDeviceCanAccessPeer": {"canAccessPeer": "sizeof(*canAccessPeer)"},
    "cudaGraphicsUnregisterResource": {"resource": "sizeof(*resource)"},
    "cudaGraphicsResourceSetMapFlags": {"resource": "sizeof(*resource)"},
    "cudaGraphicsMapResources": {"resources": "count * sizeof(*resources)", "stream": "sizeof(*stream)"},
    "cudaGraphicsUnmapResources": {"resources": "count * sizeof(*resources)", "stream": "sizeof(*stream)"},
    "cudaGraphicsResourceGetMappedPointer": {"devPtr": "sizeof(*devPtr)", "size": "sizeof(*size)", "resource": "sizeof(*resource)"},
    "cudaGraphicsSubResourceGetMappedArray": {"array": "sizeof(*array)", "resource": "sizeof(*resource)"},
    "cudaGraphicsResourceGetMappedMipmappedArray": {"mipmappedArray": "sizeof(*mipmappedArray)", "resource": "sizeof(*resource)"},
    "cudaBindTexture": {"offset": "sizeof(*offset)", "texref": "sizeof(*texref)", "devPtr": "size", "desc": "sizeof(*desc)"},
    "cudaBindTexture2D": {"offset": "sizeof(*offset)", "texref": "sizeof(*texref)", "devPtr": "pitch * height", "desc": "sizeof(*desc)"},
    "cudaBindTextureToArray": {"texref": "sizeof(*texref)", "array": "sizeof(*array)", "desc": "sizeof(*desc)"},
    "cudaBindTextureToMipmappedArray": {"texref": "sizeof(*texref)", "mipmappedArray": "sizeof(*mipmappedArray)", "desc": "sizeof(*desc)"},
    "cudaUnbindTexture": {"texref": "sizeof(*texref)"},
    "cudaGetTextureAlignmentOffset": {"offset": "sizeof(*offset)", "texref": "sizeof(*texref)"},
    "cudaGetTextureReference": {"texref": "sizeof(*texref)", "symbol": "-1"},
    "cudaBindSurfaceToArray": {"surfref": "sizeof(*surfref)", "array": "sizeof(*array)", "desc": "sizeof(*desc)"},
    "cudaGetSurfaceReference": {"surfref": "sizeof(*surfref)", "symbol": "-1"},
    "cudaGetChannelDesc": {"desc": "sizeof(*desc)", "array": "sizeof(*array)"},
    "cudaCreateChannelDesc": {"desc": "sizeof(*desc)"},
    "cudaCreateTextureObject": {"pTexObject": "sizeof(*pTexObject)", "pResDesc": "sizeof(*pResDesc)", "pTexDesc": "sizeof(*pTexDesc)", "pResViewDesc": "sizeof(*pResViewDesc)"},
    "cudaDestroyTextureObject": {"texObject": "sizeof(*texObject)"},
    "cudaGetTextureObjectResourceDesc": {"pResDesc": "sizeof(*pResDesc)", "texObject": "sizeof(*texObject)"},
    "cudaGetTextureObjectTextureDesc": {"pTexDesc": "sizeof(*pTexDesc)", "texObject": "sizeof(*texObject)"},
    "cudaGetTextureObjectResourceViewDesc": {"pResViewDesc": "sizeof(*pResViewDesc)", "texObject": "sizeof(*texObject)"},
    "cudaCreateSurfaceObject": {"pSurfObject": "sizeof(*pSurfObject)", "pResDesc": "sizeof(*pResDesc)"},
    "cudaDestroySurfaceObject": {"surfObject": "sizeof(*surfObject)"},
    "cudaGetSurfaceObjectResourceDesc": {"pResDesc": "sizeof(*pResDesc)", "surfObject": "sizeof(*surfObject)"},
    "cudaDriverGetVersion": {"driverVersion": "sizeof(*driverVersion)"},
    "cudaRuntimeGetVersion": {"runtimeVersion": "sizeof(*runtimeVersion)"},
    "cudaGraphCreate": {"pGraph": "sizeof(*pGraph)"},
    "cudaGraphAddKernelNode": {"pGraphNode": "sizeof(*pGraphNode)", "pDependencies": "numDependencies * sizeof(*pDependencies)", "pNodeParams": "sizeof(*pNodeParams)"},
    "cudaGraphKernelNodeGetParams": {"pNodeParams": "sizeof(*pNodeParams)"},
    "cudaGraphKernelNodeSetParams": {"pNodeParams": "sizeof(*pNodeParams)"},
    "cudaGraphKernelNodeCopyAttributes": {"hSrc": "sizeof(*hSrc)", "hDst": "sizeof(*hDst)"},
    "cudaGraphKernelNodeGetAttribute": {"value_out": "sizeof(*value_out)"},
    "cudaGraphKernelNodeSetAttribute": {"value": "sizeof(*value)"},
    "cudaGraphAddMemcpyNode": {"pGraphNode": "sizeof(*pGraphNode)", "pDependencies": "numDependencies * sizeof(*pDependencies)", "pCopyParams": "sizeof(*pCopyParams)"},
    "cudaGraphAddMemcpyNodeToSymbol": {"pGraphNode": "sizeof(*pGraphNode)", "pDependencies": "numDependencies * sizeof(*pDependencies)", "symbol": "-1", "src": "count"},
    "cudaGraphAddMemcpyNodeFromSymbol": {"pGraphNode": "sizeof(*pGraphNode)", "pDependencies": "numDependencies * sizeof(*pDependencies)", "dst": "count", "symbol": "-1"},
    "cudaGraphAddMemcpyNode1D": {"pGraphNode": "sizeof(*pGraphNode)", "pDependencies": "numDependencies * sizeof(*pDependencies)", "dst": "count", "src": "count"},
    "cudaGraphMemcpyNodeGetParams": {"pNodeParams": "sizeof(*pNodeParams)"},
    "cudaGraphMemcpyNodeSetParams": {"pNodeParams": "sizeof(*pNodeParams)"},
    "cudaGraphMemcpyNodeSetParamsToSymbol": {"symbol": "-1", "src": "count"},
    "cudaGraphMemcpyNodeSetParamsFromSymbol": {"dst": "count", "symbol": "-1"},
    "cudaGraphMemcpyNodeSetParams1D": {"dst": "count", "src": "count"},
    "cudaGraphAddMemsetNode": {"pGraphNode": "sizeof(*pGraphNode)", "pDependencies": "numDependencies * sizeof(*pDependencies)", "pMemsetParams": "sizeof(*pMemsetParams)"},
    "cudaGraphMemsetNodeGetParams": {"pNodeParams": "sizeof(*pNodeParams)"},
    "cudaGraphMemsetNodeSetParams": {"pNodeParams": "sizeof(*pNodeParams)"},
    "cudaGraphAddHostNode": {"pGraphNode": "sizeof(*pGraphNode)", "pDependencies": "numDependencies * sizeof(*pDependencies)", "pNodeParams": "sizeof(*pNodeParams)"},
    "cudaGraphHostNodeGetParams": {"pNodeParams": "sizeof(*pNodeParams)"},
    "cudaGraphHostNodeSetParams": {"pNodeParams": "sizeof(*pNodeParams)"},
    "cudaGraphAddChildGraphNode": {"pGraphNode": "sizeof(*pGraphNode)", "pDependencies": "numDependencies * sizeof(*pDependencies)", "childGraph": "sizeof(*childGraph)"},
    "cudaGraphChildGraphNodeGetGraph": {"pGraph": "sizeof(*pGraph)"},
    "cudaGraphAddEmptyNode": {"pGraphNode": "sizeof(*pGraphNode)", "pDependencies": "numDependencies * sizeof(*pDependencies)"},
    "cudaGraphAddEventRecordNode": {"pGraphNode": "sizeof(*pGraphNode)", "pDependencies": "numDependencies * sizeof(*pDependencies)", "event": "sizeof(*event)"},
    "cudaGraphEventRecordNodeGetEvent": {"event_out": "sizeof(*event_out)"},
    "cudaGraphEventRecordNodeSetEvent": {"event": "sizeof(*event)"},
    "cudaGraphAddEventWaitNode": {"pGraphNode": "sizeof(*pGraphNode)", "pDependencies": "numDependencies * sizeof(*pDependencies)", "event": "sizeof(*event)"},
    "cudaGraphEventWaitNodeGetEvent": {"event_out": "sizeof(*event_out)"},
    "cudaGraphEventWaitNodeSetEvent": {"event": "sizeof(*event)"},
    "cudaGraphAddExternalSemaphoresSignalNode": {"pGraphNode": "sizeof(*pGraphNode)", "pDependencies": "numDependencies * sizeof(*pDependencies)", "nodeParams": "sizeof(*nodeParams)"},
    "cudaGraphExternalSemaphoresSignalNodeGetParams": {"params_out": "sizeof(*params_out)"},
    "cudaGraphExternalSemaphoresSignalNodeSetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cudaGraphAddExternalSemaphoresWaitNode": {"pGraphNode": "sizeof(*pGraphNode)", "pDependencies": "numDependencies * sizeof(*pDependencies)", "nodeParams": "sizeof(*nodeParams)"},
    "cudaGraphExternalSemaphoresWaitNodeGetParams": {"params_out": "sizeof(*params_out)"},
    "cudaGraphExternalSemaphoresWaitNodeSetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cudaGraphAddMemAllocNode": {"pGraphNode": "sizeof(*pGraphNode)", "pDependencies": "numDependencies * sizeof(*pDependencies)", "nodeParams": "sizeof(*nodeParams)"},
    "cudaGraphMemAllocNodeGetParams": {"params_out": "sizeof(*params_out)"},
    "cudaGraphAddMemFreeNode": {"pGraphNode": "sizeof(*pGraphNode)", "pDependencies": "numDependencies * sizeof(*pDependencies)", "dptr": "sizeof(*dptr)"},
    "cudaGraphMemFreeNodeGetParams": {"dptr_out": "sizeof(*dptr_out)"},
    "cudaDeviceGetGraphMemAttribute": {"value": "sizeof(*value)"},
    "cudaDeviceSetGraphMemAttribute": {"value": "sizeof(*value)"},
    "cudaGraphClone": {"pGraphClone": "sizeof(*pGraphClone)", "originalGraph": "sizeof(*originalGraph)"},
    "cudaGraphNodeFindInClone": {"pNode": "sizeof(*pNode)", "originalNode": "sizeof(*originalNode)", "clonedGraph": "sizeof(*clonedGraph)"},
    "cudaGraphNodeGetType": {"pType": "sizeof(*pType)"},
    "cudaGraphGetNodes": {"nodes": "numNodes * sizeof(*nodes)", "numNodes": "sizeof(*numNodes)"},
    "cudaGraphGetRootNodes": {"pRootNodes": "pNumRootNodes * sizeof(*pRootNodes)", "pNumRootNodes": "sizeof(*pNumRootNodes)"},
    "cudaGraphGetEdges": {"from": "numEdges * sizeof(*from)", "to": "numEdges * sizeof(*to)", "numEdges": "sizeof(*numEdges)"},
    "cudaGraphNodeGetDependencies": {"pDependencies": "pNumDependencies * sizeof(*pDependencies)", "pNumDependencies": "sizeof(*pNumDependencies)"},
    "cudaGraphNodeGetDependentNodes": {"pDependentNodes": "pNumDependentNodes * sizeof(*pDependentNodes)", "pNumDependentNodes": "sizeof(*pNumDependentNodes)"},
    "cudaGraphAddDependencies": {"from": "numDependencies * sizeof(*from)", "to": "numDependencies * sizeof(*to)"},
    "cudaGraphRemoveDependencies": {"from": "numDependencies * sizeof(*from)", "to": "numDependencies * sizeof(*to)"},
    "cudaGraphDestroyNode": {"node": "sizeof(*node)"},
    "cudaGraphInstantiate": {"pGraphExec": "sizeof(*pGraphExec)", "pErrorNode": "sizeof(*pErrorNode)", "pLogBuffer": "bufferSize"},
    "cudaGraphInstantiateWithFlags": {"pGraphExec": "sizeof(*pGraphExec)"},
    "cudaGraphExecKernelNodeSetParams": {"pNodeParams": "sizeof(*pNodeParams)"},
    "cudaGraphExecMemcpyNodeSetParams": {"pNodeParams": "sizeof(*pNodeParams)"},
    "cudaGraphExecMemcpyNodeSetParamsToSymbol": {"symbol": "-1", "src": "count"},
    "cudaGraphExecMemcpyNodeSetParamsFromSymbol": {"dst": "count", "symbol": "-1"},
    "cudaGraphExecMemcpyNodeSetParams1D": {"dst": "count", "src": "count"},
    "cudaGraphExecMemsetNodeSetParams": {"pNodeParams": "sizeof(*pNodeParams)"},
    "cudaGraphExecHostNodeSetParams": {"pNodeParams": "sizeof(*pNodeParams)"},
    "cudaGraphExecChildGraphNodeSetParams": {"childGraph": "sizeof(*childGraph)"},
    "cudaGraphExecEventRecordNodeSetEvent": {"event": "sizeof(*event)"},
    "cudaGraphExecEventWaitNodeSetEvent": {"event": "sizeof(*event)"},
    "cudaGraphExecExternalSemaphoresSignalNodeSetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cudaGraphExecExternalSemaphoresWaitNodeSetParams": {"nodeParams": "sizeof(*nodeParams)"},
    "cudaGraphExecUpdate": {"hErrorNode_out": "sizeof(*hErrorNode_out)", "updateResult_out": "sizeof(*updateResult_out)"},
    "cudaGraphUpload": {"graphExec": "sizeof(*graphExec)", "stream": "sizeof(*stream)"},
    "cudaGraphLaunch": {"graphExec": "sizeof(*graphExec)", "stream": "sizeof(*stream)"},
    "cudaGraphExecDestroy": {"graphExec": "sizeof(*graphExec)"},
    "cudaGraphDestroy": {"graph": "sizeof(*graph)"},
    "cudaGraphDebugDotPrint": {"path": "strlen(path) + 1"},
    "cudaUserObjectCreate": {"object_out": "sizeof(*object_out)", "destroy": "sizeof(*destroy)"},
    "cudaUserObjectRetain": {"object": "sizeof(*object)"},
    "cudaUserObjectRelease": {"object": "sizeof(*object)"},
    "cudaGraphRetainUserObject": {"object": "sizeof(*object)"},
    "cudaGraphReleaseUserObject": {"object": "sizeof(*object)"},
    "cudaGetDriverEntryPoint": {"symbol": "strlen(symbol) + 1", "funcPtr": "sizeof(*funcPtr)"},
    "cudaGetExportTable": {"ppExportTable": "sizeof(*ppExportTable)", "pExportTableId": "sizeof(*pExportTableId)"},
    "cudaGetFuncBySymbol": {"functionPtr": "sizeof(*functionPtr)", "symbolPtr": "sizeof(*symbolPtr)"},
    # cuda nvml api
    "nvmlDeviceGetAttributes_v2": {"attributes": "sizeof(*attributes)"},
}


def getScalarSize(function, param):
    param_name = param.name
    if param.type.ptr_to.format() == "void" or param.type.ptr_to.format() == "const void":
        for p in function.parameters:
            if isinstance(p.type, Type):
                if p.type.format().endswith("cudaDataType"):
                    p_name = p.name
                    if p_name.endswith("Type") or p_name.endswith("type"):
                        p_name = p_name[:-4]
                        if param_name in p_name:
                            return f"sizeofType({p.name})"
        return "0"
    else:
        return f"sizeof(*{param_name})"


def getPointerSizes(function, param):
    if function.name.format() in pointer_sizes:
        if param.name in pointer_sizes[function.name.format()]:
            return pointer_sizes[function.name.format()][param.name]
    return "0"


def calculate_pointer_sizes(function, param):
    param_name = param.name
    pointer_vars = [
        "a",
        "b",
        "c",
        "s",
        "x1",
        "y1",
        "d1",
        "d2",
        "x",
        "y",
        "A",
        "B",
        "C",
        "P",
        "AP",
        "src",
        "dst",
        "srcHost",
        "dstHost",
        "param",
        "result",
        "alpha",
        "ptr",
        "workspace",
        "devicePtr",
        "hostPtr",
        "devPtr",
        "data",
        "symbol",
        "dataSizes",
        "attributes",
        "options",
    ]
    array_vars = [
        "A",
        "B",
        "C",
        "Aarray",
        "Barray",
        "Carray",
        "TauArray",
        "Ainv",
    ]
    if isinstance(param.type, Pointer):
        if param_name not in pointer_vars:
            if param.type.ptr_to.format() == "void" or param.type.ptr_to.format() == "const void":
                return "0"
            else:
                return f"sizeof(*{param.name})"
        else:
            if param_name in ["a", "b", "c", "s", "x1", "y1", "d1", "d2"]:
                return getScalarSize(function, param)
            else:
                return getPointerSizes(function, param)


def find_datatype_param(function, param_name):
    """
    查找参数对应的cudaDataType参数名
    """
    type_param_candidates = [param_name + "Type", param_name.lower() + "Type", param_name + "_type", param_name.lower() + "_type"]
    for param in function.parameters:
        if param.name in type_param_candidates and param.type.format() == "cudaDataType":
            return param.name
    return None


# Helper function used in the implementation
def has_param(name, param_names):
    return name in param_names


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
        f.write('extern "C" void mem2server(RpcClient *client, void **serverPtr, void *clientPtr, ssize_t size);\n')
        f.write('extern "C" void mem2client(RpcClient *client, void *clientPtr, ssize_t size);\n')
        f.write("void *get_so_handle(const std::string &so_file);\n")
        if header_file.endswith("cublas_api.h"):
            f.write("int sizeofType(cudaDataType type);\n")
        # 写入被 Hook 的函数实现
        if hasattr(parsed_header, "namespace") and hasattr(parsed_header.namespace, "functions"):
            function_map[header_file] = []
            is_first_function = True
            for function in parsed_header.namespace.functions:
                if function.inline:
                    continue
                function_name = function.name.format()
                if function_name in INLINE_FUNCTIONS:
                    continue
                if function_name not in MANUAL_FUNCTIONS:
                    if not is_first_function:
                        f.write("\n")
                    is_first_function = False
                    return_type = format_return_type_name(function.return_type)
                    # 函数参数列表
                    params = ", ".join([format_parameter(param) for param in function.parameters])
                    param_names = ", ".join([param.name for param in function.parameters])
                    if return_type.endswith("*"):
                        f.write(f'extern "C" {return_type}{function_name}({params}) {{\n')
                    else:
                        f.write(f'extern "C" {return_type} {function_name}({params}) {{\n')
                    f.write("#ifdef DEBUG\n")
                    f.write(f'    std::cout << "Hook: {function_name} called" << std::endl;\n')
                    f.write("#endif\n")

                    f.write(f"    RpcClient *client = rpc_get_client();\n")
                    f.write(f"    if(client == nullptr) {{\n")
                    f.write(f'        std::cerr << "Failed to get rpc client" << std::endl;\n')
                    f.write(f"        exit(1);\n")
                    f.write(f"    }}\n")
                    f.write(f"    rpc_prepare_request(client, RPC_mem2server);\n")

                    # 在rpc_get_client前调用mem2server
                    for param in function.parameters:
                        handle_param(function, param, f, True, 0)
                    f.write(f"    void *end_flag = (void *)0xffffffff;\n")
                    f.write(f"    if(client->iov_send2_count > 0) {{\n")
                    f.write(f"        rpc_write(client, &end_flag, sizeof(end_flag));\n")
                    f.write(f"        if(rpc_submit_request(client) != 0) {{\n")
                    f.write(f'            std::cerr << "Failed to submit request" << std::endl;\n')
                    f.write(f"            rpc_release_client(client);\n")
                    f.write(f"            exit(1);\n")
                    f.write(f"        }}\n")
                    f.write(f"    }}\n")

                    # 如果函数的范围类型不是void，则需要定义一个变量来保存函数的返回值
                    if return_type == "const char *":
                        f.write(f"    char *_{function_name}_result = nullptr;\n")
                    elif return_type != "void":
                        if return_type.endswith("*"):
                            f.write(f"    {return_type}_result = nullptr;\n")
                        else:
                            f.write(f"    {return_type} _result;\n")

                    f.write(f"    rpc_prepare_request(client, RPC_{function_name});\n")

                    # 在rpc_prepare_request后,rpc_submit_request前调rpc_read/rpc_write
                    for param in function.parameters:
                        handle_param(function, param, f, True, 1)

                    if return_type == "const char *":
                        f.write(f"    rpc_read(client, &_{function_name}_result, 0, true);\n")
                    elif return_type != "void":
                        f.write(f"    rpc_read(client, &_result, sizeof(_result));\n")
                    f.write(f"    if(rpc_submit_request(client) != 0) {{\n")
                    f.write(f'        std::cerr << "Failed to submit request" << std::endl;\n')
                    f.write(f"        rpc_release_client(client);\n")
                    f.write(f"        exit(1);\n")
                    f.write(f"    }}\n")

                    # 在rpc_submit_request后,rpc_free_client前针对参数进行一些处理
                    for param in function.parameters:
                        handle_param(function, param, f, True, 2)

                    f.write(f"    rpc_prepare_request(client, RPC_mem2client);\n")
                    for param in function.parameters:
                        handle_param(function, param, f, True, 3)

                    f.write(f"    if(client->iov_read2_count > 0) {{\n")
                    f.write(f"        rpc_write(client, &end_flag, sizeof(end_flag));\n")
                    f.write(f"        if(rpc_submit_request(client) != 0) {{\n")
                    f.write(f'            std::cerr << "Failed to submit request" << std::endl;\n')
                    f.write(f"            rpc_release_client(client);\n")
                    f.write(f"            exit(1);\n")
                    f.write(f"        }}\n")
                    f.write(f"    }}\n")

                    f.write(f"    rpc_free_client(client);\n")

                    if return_type == "const char *":
                        f.write(f"    return _{function_name}_result;\n")
                    elif return_type != "void":
                        f.write(f"    return _result;\n")
                    else:
                        f.write(f"    return;\n")
                    f.write("}\n")

                # 将函数名和函数原型添加到 function_map 中
                function_map[header_file].append(function)
    f.close()
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
            is_first_function = True
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
                if not is_first_function:
                    f.write("\n")
                is_first_function = False
                f.write(f"int handle_{function_name}(void *args0) {{\n")
                f.write("#ifdef DEBUG\n")
                f.write(f'    std::cout << "Handle function {function_name} called" << std::endl;\n')
                f.write("#endif\n")
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
                f.write("}\n")


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
        f.write("LDFLAGS = -ldl -lpthread -luuid\n\n")

        # 定义 hook.so 的编译参数和链接参数
        f.write("# Compilation flags for hook.so\n")
        f.write("HOOK_CXXFLAGS = $(CXXFLAGS)\n")
        f.write("HOOK_LDFLAGS = $(LDFLAGS) -shared\n\n")

        # 定义 server 的编译参数和链接参数
        f.write("# Compilation flags for server\n")
        f.write("SERVER_CXXFLAGS = $(CXXFLAGS)\n")
        f.write("SERVER_LDFLAGS = $(LDFLAGS) -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcuda -lnvidia-ml -lcublas -Wl,-rpath,/usr/local/cuda/lib64\n\n")

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
        f.write("    ../server.cpp\n\n")

        # 定义目标文件列表
        f.write("HOOK_OBJS = $(HOOK_SRCS:.cpp=.o)\n")
        f.write("SERVER_OBJS = $(SERVER_SRCS:.cpp=.o)\n\n")

        # 定义 all 目标
        f.write("all: hook.so server\n\n")

        # 编译 hook.so
        f.write("hook.so: $(HOOK_OBJS)\n")
        f.write("\t$(CXX) $(HOOK_CXXFLAGS) -o $@ $^ $(HOOK_LDFLAGS)\n\n")

        # 编译 server
        f.write("server: $(SERVER_OBJS)\n")
        f.write("\t$(CXX) $(SERVER_CXXFLAGS) -o $@ $^ $(SERVER_LDFLAGS)\n\n")

        # 清理规则
        f.write("clean:\n")
        f.write("\trm -f $(HOOK_OBJS) $(SERVER_OBJS) hook.so server\n\n")

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
