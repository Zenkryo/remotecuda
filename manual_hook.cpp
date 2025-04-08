#include <iostream>
#include <map>
#include <unordered_map>
#include <set>
#include <vector>
#include <cstdlib>
#include <sys/mman.h>
#include <mutex>
#include <memory>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include "nvml.h"
#include "gen/hook_api.h"
#include "rpc.h"
#include "hidden_api.h"

// Function to parse a PTX string and fill funcinfos into a dynamically
// allocated array
#define MAX_SYMBOL_NAME 2560
#define MAX_ARGS 64
#define MAX_REG_NAME 128
// 参数类型枚举
typedef enum {
    PARAM_B8,     // 8-bit data
    PARAM_B16,    // 16-bit data
    PARAM_B32,    // 32-bit data
    PARAM_B64,    // 64-bit data
    PARAM_S8,     // 8-bit signed integer
    PARAM_S16,    // 16-bit signed integer
    PARAM_S32,    // 32-bit signed integer
    PARAM_S64,    // 64-bit signed integer
    PARAM_U8,     // 8-bit unsigned integer
    PARAM_U16,    // 16-bit unsigned integer
    PARAM_U32,    // 32-bit unsigned integer
    PARAM_U64,    // 64-bit unsigned integer
    PARAM_F16,    // 16-bit floating point
    PARAM_F32,    // 32-bit floating point
    PARAM_F64,    // 64-bit floating point
    PARAM_F16X2,  // Two 16-bit floating points
    PARAM_V2F32,  // 2-element vector of 32-bit floats
    PARAM_V4F32,  // 4-element vector of 32-bit floats
    PARAM_V2F64,  // 2-element vector of 64-bit floats
    PARAM_V4F64,  // 4-element vector of 64-bit floats
    PARAM_PRED,   // Predicate (boolean)
    PARAM_UNKNOWN // Unknown type
} ParamType;

// 参数信息结构体
typedef struct {
    char name[MAX_SYMBOL_NAME];  // 参数明
    ParamType type;              // 参数类型
    int size;                    // 参数大小
    int is_array;                // 是否是数组
    int array_size;              // 数组长度
    int is_pointer;              // 是否是指针
    char reg_name[MAX_REG_NAME]; // 寄存器名
    void *ptr;                   // 参数指针,指向实际传递的参数
} ParamInfo;

// 函数信息结构体
typedef struct {
    void *fat_cubin;            // 函数所在的fat cubin
    void *fun_ptr;              // 函数指针
    char name[MAX_SYMBOL_NAME]; // 函数名
    ParamInfo params[MAX_ARGS]; // 参数信息
    int param_count;            // 参数个数
} FuncInfo;

std::vector<FuncInfo> funcinfos;

// 映射客户端主机内存地址到服务器主机内存地址
std::map<void *, std::pair<void *, size_t>> cs_host_mems;

// 映射客户端统一内存地址到服务器统一内存地址
std::map<void *, std::pair<void *, size_t>> cs_union_mems;

// 通过cuMemMap映射的设备内存，客户端和服务器有不同的指针
std::map<CUdeviceptr, std::pair<CUdeviceptr, size_t>> cs_dev_mems;

// 设备端内存指针（不包括上面的cs_dev_mems）
std::map<void *, size_t> server_dev_mems;

// 映射客户端内存保留地址到服务器内存保留地址
std::map<CUdeviceptr, CUdeviceptr> cs_reserve_mems;

// 服务器主机内存句柄
std::set<CUmemGenericAllocationHandle> host_handles;

void *getHookFunc(const char *symbol);

// 取得客户端主机内存地址对应的服务器主机内存地址
void *getServerHostPtr(void *ptr) {
    if(cs_host_mems.find(ptr) != cs_host_mems.end()) {
        return cs_host_mems[ptr].first;
    }
    return nullptr;
}

void *getUnionPtr(void *ptr) {
    if(cs_union_mems.find(ptr) != cs_union_mems.end()) {
        return cs_union_mems[ptr].first;
    }
    return nullptr;
}

void *getServerDevPtr(void *ptr) {
    if(cs_dev_mems.find((CUdeviceptr)ptr) != cs_dev_mems.end()) {
        return (void *)cs_dev_mems[(CUdeviceptr)ptr].first;
    }
    if(server_dev_mems.find(ptr) != server_dev_mems.end()) {
        return ptr;
    }
    return nullptr;
}

void freeDevPtr(void *ptr) {
    if(cs_union_mems.find(ptr) != cs_union_mems.end()) {
        free(ptr);
        cs_union_mems.erase(ptr);
    } else if(cs_dev_mems.find((CUdeviceptr)ptr) != cs_dev_mems.end()) {
        cs_dev_mems.erase((CUdeviceptr)ptr);
    } else if(server_dev_mems.find(ptr) != server_dev_mems.end()) {
        server_dev_mems.erase(ptr);
    }
}

// 获取参数类型
static ParamType get_param_type(const char *line) {
    if(strstr(line, ".b8"))
        return PARAM_B8;
    if(strstr(line, ".b16"))
        return PARAM_B16;
    if(strstr(line, ".s8"))
        return PARAM_S8;
    if(strstr(line, ".s16"))
        return PARAM_S16;
    if(strstr(line, ".s32"))
        return PARAM_S32;
    if(strstr(line, ".s64"))
        return PARAM_S64;
    if(strstr(line, ".u8"))
        return PARAM_U8;
    if(strstr(line, ".u16"))
        return PARAM_U16;
    if(strstr(line, ".f16"))
        return PARAM_F16;
    if(strstr(line, ".f16x2"))
        return PARAM_F16X2;
    if(strstr(line, ".v2.f64"))
        return PARAM_V2F64;
    if(strstr(line, ".v4.f64"))
        return PARAM_V4F64;
    if(strstr(line, ".b32"))
        return PARAM_B32;
    if(strstr(line, ".b64"))
        return PARAM_B64;
    if(strstr(line, ".u64"))
        return PARAM_U64;
    if(strstr(line, ".u32"))
        return PARAM_U32;
    if(strstr(line, ".f32"))
        return PARAM_F32;
    if(strstr(line, ".f64"))
        return PARAM_F64;
    if(strstr(line, ".v2.f32"))
        return PARAM_V2F32;
    if(strstr(line, ".v4.f32"))
        return PARAM_V4F32;
    if(strstr(line, ".pred"))
        return PARAM_PRED;
    return PARAM_UNKNOWN;
}

// 获取参数大小
static int get_type_size(ParamType type) {
    switch(type) {
    case PARAM_B8:
        return 1;
    case PARAM_B16:
        return 2;
    case PARAM_B32:
        return 4;
    case PARAM_B64:
        return 8;
    case PARAM_U32:
        return 4;
    case PARAM_S8:
        return 1;
    case PARAM_S16:
        return 2;
    case PARAM_S32:
        return 4;
    case PARAM_S64:
        return 8;
    case PARAM_U8:
        return 1;
    case PARAM_U16:
        return 2;
    case PARAM_F16:
        return 2;
    case PARAM_F16X2:
        return 4;
    case PARAM_PRED:
        return 1;
    case PARAM_U64:
        return 8;
    case PARAM_F32:
        return 4;
    case PARAM_F64:
        return 8;
    case PARAM_V2F32:
        return 2 * sizeof(float); // 2 element vector of 32-bit floats = 8 bytes
    case PARAM_V4F32:
        return 4 * sizeof(float); // 4 element vector of 32-bit floats = 16 bytes
    case PARAM_V2F64:
        return 2 * sizeof(double); // 2 element vector of 64-bit floats = 16 bytes
    case PARAM_V4F64:
        return 32;
    default:
        return 0;
    }
}

static void parse_ptx_string(void *fatCubin, const char *ptx_string, unsigned long long ptx_len) {
    // 跳过开头的\0字符
    while(ptx_len > 0 && *ptx_string == '\0') {
        ptx_string++;
        ptx_len--;
    }

    FuncInfo current_func = {0};
    char line[1024];
    const char *ptr = ptx_string;
    const char *end = ptx_string + ptx_len;
    int status = 0;
    while(ptr < end) {
        // 读取一行
        const char *line_end = strchr(ptr, '\n');
        if(!line_end)
            line_end = end;
        int line_len = line_end - ptr;
        if(line_len >= sizeof(line))
            line_len = sizeof(line) - 1;
        memcpy(line, ptr, line_len);
        ptr = line_end + 1;
        line[line_len] = '\0';

        // 处理函数声明行
        if(status == 0 && (strncmp(line, ".visible .entry", strlen(".visible .entry")) == 0 || strncmp(line, ".entry", strlen(".entry")) == 0)) {
            // 如果已有函数信息，可以在这里处理或输出
            memset(&current_func, 0, sizeof(FuncInfo));
            current_func.fat_cubin = fatCubin;
            char *name_start = strchr(line, '_');
            if(name_start) {
                char *name_end = strchr(name_start, '(');
                if(name_end) {
                    int name_len = name_end - name_start;
                    *name_end = '\0';
                    strncpy(current_func.name, name_start, sizeof(current_func.name) - 1);
                    status = 1; // 进入参数声明行
                }
            }
        }
        // 处理参数声明行
        else if(status == 1 && strncmp(line, ".param", strlen(".param")) == 0) {
            ParamInfo *param = &current_func.params[current_func.param_count];
            param->type = get_param_type(line);
            char *name_start = strchr(line, '_');
            if(name_start) {
                char *name_end = strpbrk(name_start, "[,");
                if(name_end) {
                    int name_len = name_end - name_start;
                    *name_end = '\0';
                    strncpy(param->name, name_start, sizeof(param->name) - 1);
                } else {
                    strncpy(param->name, name_start, sizeof(param->name) - 1);
                }
            }

            // 检查是否是数组
            char *array_start = strchr(line, '[');
            if(array_start) {
                param->is_array = 1;
                param->array_size = atoi(array_start + 1);
            }
            param->size = param->is_array ? get_type_size(param->type) * param->array_size : get_type_size(param->type);

            current_func.param_count++;
        } else if(status == 1 && strcmp(line, "{") == 0) {
            status = 2; // 进入函数体
        }
        // 处理参数加载行
        else if(status == 2 && strncmp(line, "ld.param.u64", strlen("ld.param.u64")) == 0) {
            char reg_name[MAX_REG_NAME] = {0};
            char param_name[MAX_SYMBOL_NAME] = {0};
            if(sscanf(line, "ld.param.u64 %[^,], [%[^]]]", reg_name, param_name) == 2) {
                // 查找对应的参数并记录寄存器名
                for(int i = 0; i < current_func.param_count; i++) {
                    if(strstr(param_name, current_func.params[i].name)) {
                        strncpy(current_func.params[i].reg_name, reg_name, sizeof(current_func.params[i].reg_name) - 1);
                        break;
                    }
                }
            }
        }
        // 处理指针转换行
        else if(status == 2 && strncmp(line, "cvta.to", strlen("cvta.to")) == 0) {
            char src_reg[MAX_REG_NAME] = {0};
            // cvta.to.global.u64 %rd3, %rd1;
            char *semicolon = strchr(line, ';');
            if(semicolon) {
                char *last_space = strrchr(line, ' ');
                if(last_space && last_space < semicolon) {
                    strncpy(src_reg, last_space + 1, semicolon - last_space - 1);
                    src_reg[semicolon - last_space - 1] = '\0';
                }
            }
            // 查找使用该寄存器的参数并标记为指针
            for(int i = 0; i < current_func.param_count; i++) {
                if(strcmp(current_func.params[i].reg_name, src_reg) == 0) {
                    current_func.params[i].is_pointer = 1;
                    break;
                }
            }
        } else if(status == 2 && strcmp(line, "}") == 0) {
            status = 0; // 退出函数体
            if(current_func.fat_cubin != nullptr) {
                funcinfos.push_back(current_func);
#ifdef DEBUG
                printf("==== function: %s\n", current_func.name);
                for(int i = 0; i < current_func.param_count; i++) {
                    printf("  %d: name       %s\n", i, current_func.params[i].name);
                    printf("      type       %d\n", current_func.params[i].type);
                    printf("      size       %d\n", current_func.params[i].size);
                    printf("      is array   %d\n", current_func.params[i].is_array);
                    printf("      array size %d\n", current_func.params[i].array_size);
                    printf("      is pointer %d\n", current_func.params[i].is_pointer);
                    printf("      reg name   %s\n", current_func.params[i].reg_name);
                }
#endif
            }
        }

        // 移动到下一行
        ptr = line_end + 1;
        if(ptr >= end)
            break;
    }
}

static size_t decompress(const uint8_t *input, size_t input_size, uint8_t *output, size_t output_size) {
    size_t ipos = 0, opos = 0;
    uint64_t next_nclen;  // length of next non-compressed segment
    uint64_t next_clen;   // length of next compressed segment
    uint64_t back_offset; // negative offset where redudant data is located,
                          // relative to current opos

    while(ipos < input_size) {
        next_nclen = (input[ipos] & 0xf0) >> 4;
        next_clen = 4 + (input[ipos] & 0xf);
        if(next_nclen == 0xf) {
            do {
                next_nclen += input[++ipos];
            } while(input[ipos] == 0xff);
        }

        if(memcpy(output + opos, input + (++ipos), next_nclen) == NULL) {
            fprintf(stderr, "Error copying data");
            return 0;
        }

        ipos += next_nclen;
        opos += next_nclen;
        if(ipos >= input_size || opos >= output_size) {
            break;
        }
        back_offset = input[ipos] + (input[ipos + 1] << 8);
        ipos += 2;
        if(next_clen == 0xf + 4) {
            do {
                next_clen += input[ipos++];
            } while(input[ipos - 1] == 0xff);
        }

        if(next_clen <= back_offset) {
            if(memcpy(output + opos, output + opos - back_offset, next_clen) == NULL) {
                fprintf(stderr, "Error copying data");
                return 0;
            }
        } else {
            if(memcpy(output + opos, output + opos - back_offset, back_offset) == NULL) {
                fprintf(stderr, "Error copying data");
                return 0;
            }
            for(size_t i = back_offset; i < next_clen; i++) {
                output[opos + i] = output[opos + i - back_offset];
            }
        }

        opos += next_clen;
    }
    return opos;
}

static ssize_t decompress_single_section(const uint8_t *input, uint8_t **output, size_t *output_size, struct __cudaFatCudaBinary2HeaderRec *eh, struct __cudaFatCudaBinary2EntryRec *th) {
    size_t padding;
    size_t input_read = 0;
    size_t output_written = 0;
    size_t decompress_ret = 0;
    const uint8_t zeroes[8] = {0};

    if(input == NULL || output == NULL || eh == NULL || th == NULL) {
        return 1;
    }

    uint8_t *mal = (uint8_t *)malloc(th->uncompressedBinarySize + 7);

    // add max padding of 7 bytes
    if((*output = mal) == NULL) {
        goto error;
    }

    decompress_ret = decompress(input, th->binarySize, *output, th->uncompressedBinarySize);

    // @brodey - keeping this temporarily so that we can compare the compression
    // returns
    if(decompress_ret != th->uncompressedBinarySize) {
#ifdef DEBUG
        std::cout << "failed actual decompress..." << std::endl;
#endif

        goto error;
    }
    input_read += th->binarySize;
    output_written += th->uncompressedBinarySize;

    padding = ((8 - (size_t)(input + input_read)) % 8);
    if(memcmp(input + input_read, zeroes, padding) != 0) {
        goto error;
    }
    input_read += padding;

    padding = ((8 - (size_t)th->uncompressedBinarySize) % 8);
    // Because we always allocated enough memory for one more elf_header and this
    // is smaller than the maximal padding of 7, we do not have to reallocate
    // here.
    memset(*output, 0, padding);
    output_written += padding;

    *output_size = output_written;
    return input_read;
error:
    free(*output);
    *output = NULL;
    return -1;
}

static void parseFatBinary(void *fatCubin, __cudaFatCudaBinary2Header *header) {
    char *base = (char *)(header + 1);
    long long unsigned int offset = 0;
    __cudaFatCudaBinary2EntryRec *entry = (__cudaFatCudaBinary2EntryRec *)(base);

    while(offset < header->size) {
        entry = (__cudaFatCudaBinary2EntryRec *)(base + offset);
        offset += entry->binary + entry->binarySize;
        if(!(entry->type & FATBIN_2_PTX))
            continue;
        if(entry->flags & FATBIN_FLAG_COMPRESS) {
            uint8_t *text_data = NULL;
            size_t text_data_size = 0;
            if(decompress_single_section((const uint8_t *)entry + entry->binary, &text_data, &text_data_size, header, entry) < 0) {
#ifdef DEBUG
                std::cout << "decompressing failed..." << std::endl;
#endif

                return;
            }
            parse_ptx_string(fatCubin, (char *)text_data, text_data_size);
        } else {
            parse_ptx_string(fatCubin, (char *)entry + entry->binary, entry->binarySize);
        }
    }
}

// 准备从客户端向服务器端同步内存
// 返回服务器端内存指针
extern "C" void mem2server(RpcClient *client, void **serverPtr, void *clientPtr, ssize_t size) {
#ifdef DEBUG
    std::cout << "Hook: mem2server called " << clientPtr << " " << size << std::endl;
#endif
    void *ptr = nullptr; // 服务器端内存指针(起始位置)
    size_t memSize = 0;

    if(clientPtr == nullptr) {
        return;
    }
    // 纯设备指针，不用同步内存数据
    auto it1 = cs_dev_mems.find((CUdeviceptr)clientPtr);
    if(it1 != cs_dev_mems.end()) {
        *serverPtr = (void *)it1->second.first;
        return;
    }
    auto it2 = server_dev_mems.find(clientPtr);
    if(it2 != server_dev_mems.end()) {
        *serverPtr = (void *)it2->first;
        return;
    }
    for(auto it = cs_dev_mems.begin(); it != cs_dev_mems.end(); it++) {
        if((uintptr_t)clientPtr >= (uintptr_t)it->first && (uintptr_t)clientPtr < ((uintptr_t)it->first + it->second.second)) {
            *serverPtr = (void *)((uintptr_t)it->second.first + ((uintptr_t)clientPtr - (uintptr_t)it->first));
            return;
        }
    }
    for(auto it = server_dev_mems.begin(); it != server_dev_mems.end(); it++) {
        if((uintptr_t)clientPtr >= (uintptr_t)it->first && (uintptr_t)clientPtr < ((uintptr_t)it->first + it->second)) {
            *serverPtr = clientPtr;
            return;
        }
    }
    // 函数指针，直接返回
    for(auto &func : funcinfos) {
        if(func.fun_ptr == clientPtr) {
            *serverPtr = clientPtr;
            return;
        }
    }
    // 统一内存指针
    auto it3 = cs_union_mems.find(clientPtr);
    if(it3 != cs_union_mems.end()) {
        *serverPtr = it3->second.first;
        ptr = it3->second.first;
        memSize = it3->second.second;
    } else {
        for(auto it = cs_union_mems.begin(); it != cs_union_mems.end(); it++) {
            if((uintptr_t)clientPtr >= (uintptr_t)it->first && (uintptr_t)clientPtr < ((uintptr_t)it->first + it->second.second)) {
                *serverPtr = (void *)((uintptr_t)it->second.first + ((uintptr_t)clientPtr - (uintptr_t)it->first)); // 返回的指针是带偏移的
                ptr = (void *)(uintptr_t)it->second.first;                                                          // 同步的指针是无偏移的
                memSize = it->second.second;
                break;
            }
        }
    }
    // ptr为空，表示clientPtr不是已知的设备指针，也不是统一内存指针，其可能是未知的设备内存指针，也可能是主机内存指针
    if(ptr == nullptr) {
        if(size == -1) { // size为 - 1, 表示clientPtr是设备指针，无需同步数据
            *serverPtr = clientPtr;
            return;
        }
        // 主机内存指针
        auto it = cs_host_mems.find(clientPtr);
        if(it != cs_host_mems.end()) {
            *serverPtr = it->second.first;
            ptr = it->second.first;
            memSize = it->second.second;
        } else {
            for(auto it = cs_host_mems.begin(); it != cs_host_mems.end(); it++) {
                if((uintptr_t)clientPtr >= (uintptr_t)it->first && (uintptr_t)clientPtr < ((uintptr_t)it->first + it->second.second)) {
                    *serverPtr = (void *)((uintptr_t)it->second.first + ((uintptr_t)clientPtr - (uintptr_t)it->first));
                    ptr = (void *)(uintptr_t)it->second.first;
                    memSize = it->second.second;
                    break;
                }
            }
        }
    }
    // ptr为空表示clientPtr是未知的主机内存
    if(ptr == nullptr) {
        if(size == 0) { // 大小不知道，无法同步数据
            printf("WARNING: no size info for client host memory 0x%p\n", clientPtr);
            *serverPtr = clientPtr;
            return;
        }
        // 写入null指针
        void **tmp_ptr = (void **)malloc(sizeof(ptr));
        *tmp_ptr = ptr;
        client->tmps4iov.insert(tmp_ptr);
        rpc_write(client, tmp_ptr, sizeof(*tmp_ptr));

        // 写入客户端内存数据
        rpc_write(client, clientPtr, size, true);

        // 读取服务器端内存指针
        rpc_read(client, serverPtr, sizeof(*serverPtr));
    } else {
        // 写入服务器端内存指针
        void **tmp_ptr = (void **)malloc(sizeof(ptr));
        *tmp_ptr = ptr;
        client->tmps4iov.insert(tmp_ptr);
        rpc_write(client, tmp_ptr, sizeof(*tmp_ptr));

        // 写入客户端内存数据
        rpc_write(client, clientPtr, memSize, true);
    }
    return;
}

// 准备从服务器向客户端同步内存
extern "C" void mem2client(RpcClient *client, void *clientPtr, ssize_t size) {
#ifdef DEBUG
    std::cout << "Hook: mem2client called " << clientPtr << " " << size << std::endl;
#endif
    void *ptr = nullptr; // 服务器端内存指针(起始位置)
    size_t memSize = 0;

    if(clientPtr == nullptr) {
        return;
    }
    // 纯设备指针，不用同步内存数据
    auto it1 = cs_dev_mems.find((CUdeviceptr)clientPtr);
    if(it1 != cs_dev_mems.end()) {
        return;
    }
    auto it2 = server_dev_mems.find(clientPtr);
    if(it2 != server_dev_mems.end()) {
        return;
    }
    for(auto it = cs_dev_mems.begin(); it != cs_dev_mems.end(); it++) {
        if((uintptr_t)clientPtr >= (uintptr_t)it->first && (uintptr_t)clientPtr < ((uintptr_t)it->first + it->second.second)) {
            return;
        }
    }
    for(auto it = server_dev_mems.begin(); it != server_dev_mems.end(); it++) {
        if((uintptr_t)clientPtr >= (uintptr_t)it->first && (uintptr_t)clientPtr < ((uintptr_t)it->first + it->second)) {
            return;
        }
    }
    // 函数指针，直接返回
    for(auto &func : funcinfos) {
        if(func.fun_ptr == clientPtr) {
            return;
        }
    }
    // 统一内存指针
    auto it3 = cs_union_mems.find(clientPtr);
    if(it3 != cs_union_mems.end()) {
        ptr = it3->second.first;
        memSize = it3->second.second;
    } else {
        for(auto it = cs_union_mems.begin(); it != cs_union_mems.end(); it++) {
            if((uintptr_t)clientPtr >= (uintptr_t)it->first && (uintptr_t)clientPtr < ((uintptr_t)it->first + it->second.second)) {
                clientPtr = (void *)(uintptr_t)it->first;  // 修改clientPtr到内存的起始位置
                ptr = (void *)(uintptr_t)it->second.first; // 同步的指针是无偏移的
                memSize = it->second.second;
                break;
            }
        }
    }
    // ptr为空，表示clientPtr不是已知的设备指针，也不是统一内存指针，其可能是未知的设备内存指针，也可能是主机内存指针
    if(ptr == nullptr) {
        if(size == -1) { // size为 - 1, 表示clientPtr是设备指针，无需同步数据
            return;
        }
        // 主机内存指针
        auto it = cs_host_mems.find(clientPtr);
        if(it != cs_host_mems.end()) {
            ptr = it->second.first;
            memSize = it->second.second;
        } else {
            for(auto it = cs_host_mems.begin(); it != cs_host_mems.end(); it++) {
                if((uintptr_t)clientPtr >= (uintptr_t)it->first && (uintptr_t)clientPtr < ((uintptr_t)it->first + it->second.second)) {
                    clientPtr = (void *)(uintptr_t)it->first; // 修改clientPtr到内存的起始位置
                    ptr = (void *)(uintptr_t)it->second.first;
                    memSize = it->second.second;
                    break;
                }
            }
        }
    }
    // ptr为空表示clientPtr是未知的主机内存
    if(ptr == nullptr) {
        if(size <= 0) { // 大小不知道，无法同步数据
            printf("WARNING: no size info for client host memory 0x%p\n", clientPtr);
            return;
        }
        // 写入null指针
        void **tmp_ptr = (void **)malloc(sizeof(ptr));
        *tmp_ptr = ptr;
        client->tmps4iov.insert(tmp_ptr);
        rpc_write(client, tmp_ptr, sizeof(*tmp_ptr));

        // 写入大小
        ssize_t *tmp_size = (ssize_t *)malloc(sizeof(size));
        *tmp_size = size;
        client->tmps4iov.insert(tmp_size);
        rpc_write(client, tmp_size, sizeof(*tmp_size));

        // 读取数据到客户端内存
        rpc_read(client, clientPtr, size, true);
    } else if(memSize > 0) {
        // 写入服务器端内存指针
        void **tmp_ptr = (void **)malloc(sizeof(ptr));
        *tmp_ptr = ptr;
        client->tmps4iov.insert(tmp_ptr);
        rpc_write(client, tmp_ptr, sizeof(*tmp_ptr));

        // 写入大小
        ssize_t *tmp_size = (ssize_t *)malloc(sizeof(size));
        *tmp_size = memSize;
        client->tmps4iov.insert(tmp_size);
        rpc_write(client, tmp_size, sizeof(*tmp_size));

        // 读取数据到客户端内存
        rpc_read(client, clientPtr, memSize, true);
    }
    return;
}

extern "C" cudaError_t cudaFree(void *devPtr) {
#ifdef DEBUG
    std::cout << "Hook: cudaFree called" << std::endl;
#endif

    cudaError_t _result;
    void *serverPtr = getUnionPtr(devPtr);
    if(serverPtr == nullptr) {
        serverPtr = getServerDevPtr(devPtr);
        if(serverPtr == nullptr) {
            serverPtr = devPtr;
        }
    }
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaFree);
    rpc_write(client, &serverPtr, sizeof(serverPtr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        freeDevPtr(devPtr);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaFreeHost(void *ptr) {
#ifdef DEBUG
    std::cout << "Hook: cudaFreeHost called" << std::endl;
#endif

    cudaError_t _result;
    void *serverPtr;
    serverPtr = getServerHostPtr((void *)ptr);
    if(serverPtr == nullptr) {
        return cudaErrorInvalidValue;
    }
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaFreeHost);
    rpc_write(client, &serverPtr, sizeof(serverPtr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        free(ptr);
        cs_host_mems.erase(ptr);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" const char *cudaGetErrorName(cudaError_t error) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetErrorName called" << std::endl;
#endif

    static char _cudaGetErrorName_result[1024];
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaGetErrorName);
    rpc_write(client, &error, sizeof(error));
    rpc_read(client, _cudaGetErrorName_result, 1024, true);
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _cudaGetErrorName_result;
}

extern "C" const char *cudaGetErrorString(cudaError_t error) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetErrorString called" << std::endl;
#endif

    static char _cudaGetErrorString_result[1024];
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaGetErrorString);
    rpc_write(client, &error, sizeof(error));
    rpc_read(client, _cudaGetErrorString_result, 1024, true);
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _cudaGetErrorString_result;
}

extern "C" cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetSymbolAddress called" << std::endl;
#endif

    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaGetSymbolAddress);
    rpc_read(client, devPtr, sizeof(*devPtr));
    rpc_write(client, &symbol, sizeof(symbol));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        server_dev_mems[*devPtr] = 0;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaHostAlloc called" << std::endl;
#endif

    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *pHost = malloc(size);
    if(*pHost == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    rpc_prepare_request(client, RPC_cudaHostAlloc);
    rpc_read(client, &serverPtr, sizeof(serverPtr));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        free(*pHost);
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        cs_host_mems[*pHost] = std::make_pair(serverPtr, size);
    } else {
        free(*pHost);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaLaunchKernel called" << std::endl;
#endif

    cudaError_t _result;
    FuncInfo *f = nullptr;
    for(auto &funcinfo : funcinfos) {
        if(funcinfo.fun_ptr == func) {
            f = &funcinfo;
            break;
        }
    }
    if(f == nullptr) {
        std::cerr << "Failed to find function" << std::endl;
        return cudaErrorInvalidDeviceFunction;
    }
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_mem2server);
    for(int i = 0; i < f->param_count; i++) {
        mem2server(client, &f->params[i].ptr, *((void **)args[i]), -1);
    }
    void *end_flag = (void *)0xffffffff;
    if(client->iov_send2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }

    rpc_prepare_request(client, RPC_cudaLaunchKernel);
    rpc_write(client, &func, sizeof(func));
    rpc_write(client, &gridDim, sizeof(gridDim));
    rpc_write(client, &blockDim, sizeof(blockDim));
    rpc_write(client, &sharedMem, sizeof(sharedMem));
    rpc_write(client, &stream, sizeof(stream));
    rpc_write(client, &f->param_count, sizeof(f->param_count));

    for(int i = 0; i < f->param_count; i++) {
        rpc_write(client, &f->params[i].ptr, f->params[i].size, true);
    }
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // _result = cudaDeviceSynchronize();
    rpc_prepare_request(client, RPC_mem2client);
    for(int i = 0; i < f->param_count; i++) {
        mem2client(client, *((void **)args[i]), -1);
    }
    if(client->iov_read2_count > 0) {
        rpc_write(client, &end_flag, sizeof(end_flag));
        if(rpc_submit_request(client) != 0) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_client(client);
            exit(1);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
#ifdef DEBUG
    std::cout << "Hook: cudaMalloc called" << std::endl;
#endif

    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaMalloc);
    rpc_read(client, devPtr, sizeof(*devPtr));
    rpc_write(client, &size, sizeof(size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        server_dev_mems[*devPtr] = size;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr, struct cudaExtent extent) {
#ifdef DEBUG
    std::cout << "Hook: cudaMalloc3D called" << std::endl;
#endif

    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaMalloc3D);
    rpc_read(client, pitchedDevPtr, sizeof(*pitchedDevPtr));
    rpc_write(client, &extent, sizeof(extent));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        server_dev_mems[pitchedDevPtr->ptr] = pitchedDevPtr->pitch * extent.height;
    }
    rpc_free_client(client);
    return _result;
}

// 是不是只在客户端这边做客户端主机内存和服务器主机内存的映射就够了？
extern "C" cudaError_t cudaMallocHost(void **ptr, size_t size) {
#ifdef DEBUG
    std::cout << "Hook: cudaMallocHost called" << std::endl;
#endif

    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *ptr = malloc(size);
    if(*ptr == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    rpc_prepare_request(client, RPC_cudaMallocHost);
    rpc_read(client, &serverPtr, sizeof(serverPtr));
    rpc_write(client, &size, sizeof(size));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        free(*ptr);
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        cs_host_mems[*ptr] = std::make_pair(serverPtr, size);
    } else {
        free(*ptr);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaMallocManaged called" << std::endl;
#endif

    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *devPtr = malloc(size);
    if(*devPtr == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    rpc_prepare_request(client, RPC_cudaMallocManaged);
    rpc_read(client, &serverPtr, sizeof(serverPtr));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        free(*devPtr);
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        cs_union_mems[*devPtr] = std::make_pair((void *)serverPtr, size);
    } else {
        free(*devPtr);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) {
#ifdef DEBUG
    std::cout << "Hook: cudaMallocPitch called" << std::endl;
#endif

    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cudaMallocPitch);
    rpc_read(client, devPtr, sizeof(*devPtr));
    rpc_read(client, pitch, sizeof(*pitch));
    rpc_write(client, &width, sizeof(width));
    rpc_write(client, &height, sizeof(height));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == cudaSuccess) {
        server_dev_mems[*devPtr] = *pitch * height;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream) {
#ifdef DEBUG
    std::cout << "Hook: __cudaPopCallConfiguration called" << std::endl;
#endif

    cudaError_t _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaPopCallConfiguration);
    rpc_read(client, gridDim, sizeof(*gridDim));
    rpc_read(client, blockDim, sizeof(*blockDim));
    rpc_read(client, sharedMem, sizeof(*sharedMem));
    rpc_read(client, stream, sizeof(cudaStream_t));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, struct CUstream_st *stream) {
#ifdef DEBUG
    std::cout << "Hook: __cudaPushCallConfiguration called" << std::endl;
#endif

    unsigned _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaPushCallConfiguration);
    rpc_write(client, &gridDim, sizeof(gridDim));
    rpc_write(client, &blockDim, sizeof(blockDim));
    rpc_write(client, &sharedMem, sizeof(sharedMem));
    rpc_write(client, &stream, sizeof(stream));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" void **__cudaRegisterFatBinary(void *fatCubin) {
#ifdef DEBUG
    std::cout << "Hook: __cudaRegisterFatBinary called" << std::endl;
#endif

    void **_result = nullptr;
    __cudaFatCudaBinary2 *binary;
    __cudaFatCudaBinary2Header *header;
    unsigned long long size;

    binary = (__cudaFatCudaBinary2 *)fatCubin;
    header = (__cudaFatCudaBinary2Header *)binary->text;
    size = sizeof(__cudaFatCudaBinary2Header) + header->size;

    if(*(unsigned *)fatCubin != __cudaFatMAGIC2) {
        std::cerr << "Invalid fat binary magic" << std::endl;
        return nullptr;
    }
    parseFatBinary(fatCubin, header);

    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaRegisterFatBinary);
    rpc_write(client, binary, sizeof(__cudaFatCudaBinary2));
    rpc_write(client, header, size, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
#ifdef DEBUG
    std::cout << "Hook: __cudaRegisterFatBinaryEnd called" << std::endl;
#endif

    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaRegisterFatBinaryEnd);
    rpc_write(client, &fatCubinHandle, sizeof(fatCubinHandle));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
}

extern "C" void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
#ifdef DEBUG
    std::cout << "Hook: __cudaRegisterFunction called" << std::endl;
#endif
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaRegisterFunction);
    rpc_write(client, &fatCubinHandle, sizeof(fatCubinHandle));
    rpc_write(client, &hostFun, sizeof(hostFun));
    rpc_write(client, deviceName, strlen(deviceName) + 1, true);
    rpc_write(client, &thread_limit, sizeof(thread_limit));
    uint8_t mask = 0;
    if(tid != nullptr)
        mask |= 1 << 0;
    if(bid != nullptr)
        mask |= 1 << 1;
    if(bDim != nullptr)
        mask |= 1 << 2;
    if(gDim != nullptr)
        mask |= 1 << 3;
    if(wSize != nullptr)
        mask |= 1 << 4;
    rpc_write(client, &mask, sizeof(mask));
    if(tid != nullptr) {
        rpc_write(client, tid, sizeof(uint3));
    }
    if(bid != nullptr) {
        rpc_write(client, bid, sizeof(uint3));
    }
    if(bDim != nullptr) {
        rpc_write(client, bDim, sizeof(dim3));
    }
    if(gDim != nullptr) {
        rpc_write(client, gDim, sizeof(dim3));
    }
    if(wSize != nullptr) {
        rpc_write(client, wSize, sizeof(int));
    }
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    // also memorize the host pointer function
    for(auto &funcinfo : funcinfos) {
        if(strcmp(funcinfo.name, deviceName) == 0) {
            funcinfo.fun_ptr = (void *)hostFun;
        }
    }
    rpc_free_client(client);
}

extern "C" void __cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress, char *deviceAddress, const char *deviceName, int ext, size_t size, int constant, int global) {
#ifdef DEBUG
    std::cout << "Hook: __cudaRegisterManagedVar called" << std::endl;
#endif

    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaRegisterManagedVar);
    rpc_write(client, &fatCubinHandle, sizeof(fatCubinHandle));
    rpc_write(client, deviceName, strlen(deviceName) + 1, true);
    rpc_write(client, &ext, sizeof(ext));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &constant, sizeof(constant));
    rpc_write(client, &global, sizeof(global));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }

    rpc_free_client(client);
}

extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, size_t size, int constant, int global) {
#ifdef DEBUG
    std::cout << "Hook: __cudaRegisterVar called" << std::endl;
#endif

    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaRegisterVar);
    rpc_write(client, &fatCubinHandle, sizeof(fatCubinHandle));
    rpc_write(client, &hostVar, sizeof(hostVar));
    rpc_write(client, deviceName, strlen(deviceName) + 1, true);
    rpc_write(client, &ext, sizeof(ext));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &constant, sizeof(constant));
    rpc_write(client, &global, sizeof(global));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    server_dev_mems[(void *)hostVar] = size;
    rpc_free_client(client);
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle) {
#ifdef DEBUG
    std::cout << "Hook: __cudaUnregisterFatBinary called" << std::endl;
#endif

    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaUnregisterFatBinary);
    rpc_write(client, &fatCubinHandle, sizeof(fatCubinHandle));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
}

extern "C" char __cudaInitModule(void **fatCubinHandle) {
#ifdef DEBUG
    std::cout << "Hook: __cudaInitModule called" << std::endl;
#endif

    char _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC___cudaInitModule);
    // PARAM void **fatCubinHandle
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    // PARAM void **fatCubinHandle
    rpc_free_client(client);
    return _result;
}

// CUDA Driver API (cu*)
extern "C" CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuExternalMemoryGetMappedBuffer called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuExternalMemoryGetMappedBuffer);
    rpc_read(client, devPtr, sizeof(*devPtr));
    rpc_write(client, &extMem, sizeof(extMem));
    rpc_write(client, bufferDesc, sizeof(*bufferDesc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*devPtr] = bufferDesc->size;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGetErrorName(CUresult error, const char **pStr) {
#ifdef DEBUG
    std::cout << "Hook: cuGetErrorName called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGetErrorName);
    rpc_write(client, &error, sizeof(error));
    static char _cuGetErrorName_pStr[1024];
    rpc_read(client, _cuGetErrorName_pStr, 1024, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    *pStr = _cuGetErrorName_pStr;
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGetErrorString(CUresult error, const char **pStr) {
#ifdef DEBUG
    std::cout << "Hook: cuGetErrorString called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGetErrorString);
    rpc_write(client, &error, sizeof(error));
    static char _cuGetErrorString_pStr[1024];
    rpc_read(client, _cuGetErrorString_pStr, 1024, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    *pStr = _cuGetErrorString_pStr;
    rpc_free_client(client);
    return _result;
}

#if CUDA_VERSION <= 11040
extern "C" CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
#ifdef DEBUG
    std::cout << "Hook: cuGetProcAddress called" << std::endl;
#endif

    *pfn = getHookFunc(symbol);
    if(*pfn != nullptr) {
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_NOT_FOUND;
}
#endif

extern "C" CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphicsResourceGetMappedPointer_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphicsResourceGetMappedPointer_v2);
    rpc_read(client, pDevPtr, sizeof(*pDevPtr));
    rpc_read(client, pSize, sizeof(*pSize));
    rpc_write(client, &resource, sizeof(resource));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*pDevPtr] = *pSize;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuImportExternalMemory(CUexternalMemory *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuImportExternalMemory called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuImportExternalMemory);
    rpc_read(client, extMem_out, sizeof(*extMem_out));
    rpc_write(client, memHandleDesc, sizeof(*memHandleDesc));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*extMem_out] = memHandleDesc->size;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuIpcOpenMemHandle_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuIpcOpenMemHandle_v2);
    rpc_read(client, pdptr, sizeof(*pdptr));
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*pdptr] = 0;
    }
    rpc_free_client(client);
    return _result;
}
#if CUDA_VERSION > 11040
extern "C" CUresult cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuLibraryGetGlobal called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLibraryGetGlobal);
    rpc_read(client, dptr, sizeof(*dptr));
    rpc_read(client, bytes, sizeof(*bytes));
    rpc_write(client, &library, sizeof(library));
    rpc_write(client, name, strlen(name) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr] = *bytes;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuLibraryGetManaged called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuLibraryGetManaged);
    rpc_read(client, dptr, sizeof(*dptr));
    rpc_read(client, bytes, sizeof(*bytes));
    rpc_write(client, &library, sizeof(library));
    rpc_write(client, name, strlen(name) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        void *p = malloc(*bytes);
        if(p == nullptr) {
            // TODO: free memory on server
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        cs_union_mems[p] = std::make_pair((void *)*dptr, *bytes);
    }
    rpc_free_client(client);
    return _result;
}
#endif

// 此函数用于保留一段连续的虚拟地址空间，供后续的内存映射操作使用
extern "C" CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAddressReserve called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    CUdeviceptr *serverPtr;
    // 客户端保留内存地址
    *ptr = (CUdeviceptr)mmap(NULL, size, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if(*ptr == 0) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    rpc_prepare_request(client, RPC_cuMemAddressReserve);
    rpc_read(client, serverPtr, sizeof(*serverPtr));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &alignment, sizeof(alignment));
    rpc_write(client, &addr, sizeof(addr));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        munmap((void *)*ptr, size);
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        cs_reserve_mems[*ptr] = *serverPtr;
    } else {
        munmap((void *)*ptr, size);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAlloc_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemAlloc_v2);
    rpc_read(client, dptr, sizeof(*dptr));
    rpc_write(client, &bytesize, sizeof(bytesize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr] = bytesize;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAllocHost_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *pp = malloc(bytesize);
    if(*pp == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    rpc_prepare_request(client, RPC_cuMemAllocHost_v2);
    rpc_read(client, &serverPtr, sizeof(serverPtr));
    rpc_write(client, &bytesize, sizeof(bytesize));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        free(*pp);
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        cs_host_mems[*pp] = std::make_pair(serverPtr, bytesize);
    } else {
        free(*pp);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAllocManaged called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *dptr = (CUdeviceptr)malloc(bytesize);
    if(*dptr == 0) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    rpc_prepare_request(client, RPC_cuMemAllocManaged);
    rpc_read(client, dptr, sizeof(*dptr));
    rpc_write(client, &bytesize, sizeof(bytesize));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        cs_union_mems[(void *)*dptr] = std::make_pair((void *)serverPtr, bytesize);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAllocPitch_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemAllocPitch_v2);
    rpc_read(client, dptr, sizeof(*dptr));
    rpc_read(client, pPitch, sizeof(*pPitch));
    rpc_write(client, &WidthInBytes, sizeof(WidthInBytes));
    rpc_write(client, &Height, sizeof(Height));
    rpc_write(client, &ElementSizeBytes, sizeof(ElementSizeBytes));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr] = *pPitch * Height;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemCreate called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemCreate);
    rpc_read(client, handle, sizeof(*handle));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, prop, sizeof(*prop));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
#ifdef CU_MEM_LOCATION_TYPE_HOST
        // 如果是主机内存,则记录句柄
        if(prop->location.type == CU_MEM_LOCATION_TYPE_HOST) {
            host_handles.insert(*handle);
        }
#endif
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemFreeHost(void *p) {
#ifdef DEBUG
    std::cout << "Hook: cuMemFreeHost called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    void *serverPtr;
    if(cs_host_mems.find(p) == cs_host_mems.end()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    serverPtr = cs_host_mems[p].first;
    rpc_prepare_request(client, RPC_cuMemFreeHost);
    rpc_write(client, &serverPtr, sizeof(serverPtr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        free(p);
        cs_host_mems.erase(p);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
#ifdef DEBUG
    std::cout << "Hook: cuMemGetAddressRange_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemGetAddressRange_v2);
    rpc_read(client, pbase, sizeof(*pbase));
    rpc_read(client, psize, sizeof(*psize));
    rpc_write(client, &dptr, sizeof(dptr));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*pbase] = *psize;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemHostAlloc called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *pp = malloc(bytesize);
    if(*pp == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    rpc_prepare_request(client, RPC_cuMemHostAlloc);
    rpc_read(client, &serverPtr, sizeof(serverPtr));
    rpc_write(client, &bytesize, sizeof(bytesize));
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        free(*pp);
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        cs_host_mems[*pp] = std::make_pair(serverPtr, bytesize);
    } else {
        free(*pp);
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemHostGetDevicePointer_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemHostGetDevicePointer_v2);
    rpc_read(client, pdptr, sizeof(*pdptr));
    void *serverPtr = getServerHostPtr(p);
    if(serverPtr == nullptr) {
        rpc_write(client, &p, sizeof(p));
    } else {
        rpc_write(client, &serverPtr, sizeof(serverPtr));
    }
    rpc_write(client, &Flags, sizeof(Flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*pdptr] = 0;
    }
    rpc_free_client(client);
    return _result;
}

// 此函数将物理内存块映射到预先保留的虚拟地址空间，使该内存可被GPU或CPU访问
extern "C" CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemMap called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    CUdeviceptr serverPtr;
    if(cs_reserve_mems.find(ptr) != cs_reserve_mems.end()) {
        serverPtr = cs_reserve_mems[ptr];
    } else {
        return CUDA_ERROR_INVALID_VALUE;
    }
    bool isHost = false;
    // 如何handle存在于host_handles中,则说明是在主机内存上映射
    if(host_handles.find(handle) != host_handles.end()) {
        isHost = true;
        // 在客户端实际映射主机内存到客户端保留地址
        mmap((void *)ptr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, -1, 0);
    }
    rpc_prepare_request(client, RPC_cuMemMap);
    rpc_write(client, &serverPtr, sizeof(serverPtr));
    rpc_write(client, &size, sizeof(size));
    rpc_write(client, &offset, sizeof(offset));
    rpc_write(client, &handle, sizeof(handle));
    rpc_write(client, &flags, sizeof(flags));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        if(isHost) {
            munmap((void *)ptr, size);
        }
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        if(isHost) {
            cs_host_mems[(void *)ptr] = std::make_pair((void *)serverPtr, size);
        } else {
            cs_dev_mems[ptr] = std::make_pair(serverPtr, size);
        }
        cs_reserve_mems.erase(ptr);
    } else {
        if(isHost) {
            munmap((void *)ptr, size);
        }
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData *shareData) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolImportPointer called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemPoolImportPointer);
    rpc_read(client, ptr_out, sizeof(*ptr_out));
    rpc_write(client, &pool, sizeof(pool));
    rpc_write(client, shareData, sizeof(*shareData));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        // TODO 通过拦截cuMemPoolExportPointer获取size，现在先将size置0
        server_dev_mems[(void *)*ptr_out] = 0;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
#ifdef DEBUG
    std::cout << "Hook: cuMemRelease called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuMemRelease);
    rpc_write(client, &handle, sizeof(handle));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        host_handles.erase(handle);
    }
    rpc_free_client(client);
    return _result;
}

// #if CUDA_VERSION > 11040
// extern "C" CUresult cuMemcpyBatchAsync(CUdeviceptr *dsts, CUdeviceptr *srcs, size_t *sizes, size_t count, CUmemcpyAttributes *attrs, size_t *attrsIdxs, size_t numAttrs, size_t *failIdx, CUstream hStream) {
// #ifdef DEBUG
//     std::cout << "Hook: cuMemcpyBatchAsync called" << std::endl;
// #endif

//     CUresult _result;
//     RpcClient *client = rpc_get_client();
//     if(client == nullptr) {
//         std::cerr << "Failed to get rpc client" << std::endl;
//         exit(1);
//     }
//     rpc_prepare_request(client, RPC_cuMemcpyBatchAsync);
//     CUdeviceptr *new_dsts = (CUdeviceptr *)malloc(count * sizeof(CUdeviceptr));
//     CUdeviceptr *new_srcs = (CUdeviceptr *)malloc(count * sizeof(CUdeviceptr));
//     if(new_dsts == nullptr || new_srcs == nullptr) {
//         return CUDA_ERROR_OUT_OF_MEMORY;
//     }
//     for(int i = 0; i < count; i++) {
//         if(cs_union_mems.find((void *)dsts[i]) != cs_union_mems.end()) {
//             std::pair<void *, size_t> mem = cs_union_mems[(void *)dsts[i]];
//             new_dsts[i] = (CUdeviceptr)mem.first;
//             rpc_write(client, &new_dsts[i], sizeof(new_dsts[i]));
//             // 如果是拷贝到统一内存,还需要将数据读回到客户端, 服务器端需要调用cudaPointerGetAttributes来判断指针是否是统一内存
//             rpc_read(client, (void *)dsts[i], sizes[i], true);
//         } else if(cs_dev_mems.find(dsts[i]) != cs_dev_mems.end()) {
//             new_dsts[i] = (CUdeviceptr)cs_dev_mems[dsts[i]];
//             rpc_write(client, &new_dsts[i], sizeof(new_dsts[i]));
//         } else {
//             rpc_write(client, (void *)&dsts[i], sizeof(dsts[i]));
//         }
//     }
//     for(int i = 0; i < count; i++) {
//         if(cs_union_mems.find((void *)srcs[i]) != cs_union_mems.end()) {
//             std::pair<void *, size_t> mem = cs_union_mems[(void *)srcs[i]];
//             new_srcs[i] = (CUdeviceptr)mem.first;
//             rpc_write(client, &new_srcs[i], sizeof(new_srcs[i]));
//             // 如果是从统一内存拷贝,还需要将数据写入到服务器端
//             rpc_write(client, (void *)srcs[i], sizes[i], true);
//         } else if(cs_dev_mems.find(srcs[i]) != cs_dev_mems.end()) {
//             new_srcs[i] = (CUdeviceptr)cs_dev_mems[srcs[i]];
//             rpc_write(client, &new_srcs[i], sizeof(new_srcs[i]));
//         } else {
//             rpc_write(client, (void *)&srcs[i], sizeof(srcs[i]));
//         }
//     }
//     rpc_write(client, &count, sizeof(count));
//     rpc_write(client, sizes, sizeof(*sizes) * count);
//     rpc_read(client, attrs, sizeof(*attrs));
//     rpc_read(client, attrsIdxs, sizeof(*attrsIdxs));
//     rpc_write(client, &numAttrs, sizeof(numAttrs));
//     rpc_read(client, failIdx, sizeof(*failIdx));
//     rpc_write(client, &hStream, sizeof(hStream));
//     rpc_read(client, &_result, sizeof(_result));
//     if(rpc_submit_request(client) != 0) {
//         std::cerr << "Failed to submit request" << std::endl;
//         rpc_release_client(client);
//         exit(1);
//     }
//     rpc_free_client(client);
//     return _result;
// }
// #endif

extern "C" CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleGetGlobal_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuModuleGetGlobal_v2);
    rpc_read(client, dptr, sizeof(*dptr));
    rpc_read(client, bytes, sizeof(*bytes));
    rpc_write(client, &hmod, sizeof(hmod));
    rpc_write(client, name, strlen(name) + 1, true);
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr] = *bytes;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuTexRefGetAddress_v2(CUdeviceptr *pdptr, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetAddress_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuTexRefGetAddress_v2);
    rpc_read(client, pdptr, sizeof(*pdptr));
    rpc_write(client, &hTexRef, sizeof(hTexRef));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*pdptr] = 0;
    }
    rpc_free_client(client);
    return _result;
}

extern "C" CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr *dptr_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemFreeNodeGetParams called" << std::endl;
#endif

    CUresult _result;
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_cuGraphMemFreeNodeGetParams);
    rpc_write(client, &hNode, sizeof(hNode));
    rpc_read(client, dptr_out, sizeof(*dptr_out));
    rpc_read(client, &_result, sizeof(_result));
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr_out] = 0; // TODO?
    }
    rpc_free_client(client);
    return _result;
}

// NVML (nvml*)

extern "C" const char *nvmlErrorString(nvmlReturn_t result) {
#ifdef DEBUG
    std::cout << "Hook: nvmlErrorString called" << std::endl;
#endif

    static char _nvmlErrorString_result[1024];
    RpcClient *client = rpc_get_client();
    if(client == nullptr) {
        std::cerr << "Failed to get rpc client" << std::endl;
        exit(1);
    }
    rpc_prepare_request(client, RPC_nvmlErrorString);
    rpc_write(client, &result, sizeof(result));
    rpc_read(client, _nvmlErrorString_result, 1024, true);
    if(rpc_submit_request(client) != 0) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_client(client);
        exit(1);
    }
    rpc_free_client(client);
    return _nvmlErrorString_result;
}
