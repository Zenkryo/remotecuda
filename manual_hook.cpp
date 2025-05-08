#include <iostream>
#include <map>
#include <map>
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
#include "rpc/rpc_core.h"
#include "hidden_api.h"
#include "client.h"

std::vector<FuncInfo> funcinfos;

// 映射客户端主机内存地址到服务器主机内存地址
std::map<void *, std::pair<void *, size_t>> cs_host_mems;

// 映射客户端主机内存地址到服务器主机内存地址(临时内存)
std::map<void *, std::pair<void *, size_t>> cs_host_tmp_mems;

// 映射客户端统一内存地址到服务器统一内存地址
std::map<void *, std::pair<void *, size_t>> cs_union_mems;

// 设备端内存指针（不包括上面的cs_dev_mems）
std::map<void *, size_t> server_dev_mems;

// 服务器主机内存句柄
std::set<CUmemGenericAllocationHandle> host_handles;

void *getHookFunc(const char *symbol);

// 取得客户端主机内存地址对应的服务器主机内存地址
void *getServerHostPtr(void *clientPtr) {
    auto it1 = cs_union_mems.find(clientPtr);
    if(it1 != cs_union_mems.end()) {
        return it1->second.first;
    }
    auto it2 = cs_host_mems.find(clientPtr);
    if(it2 != cs_host_mems.end()) {
        return it2->second.first;
    }
    auto it3 = cs_host_tmp_mems.find(clientPtr);
    if(it3 != cs_host_tmp_mems.end()) {
        return it3->second.first;
    }
    return nullptr;
}

void *getClientHostPtr(void *serverPtr) {
    for(auto it = cs_host_mems.begin(); it != cs_host_mems.end(); it++) {
        if(it->second.first == serverPtr) {
            return it->first;
        }
    }
    for(auto it = cs_host_tmp_mems.begin(); it != cs_host_tmp_mems.end(); it++) {
        if(it->second.first == serverPtr) {
            return it->first;
        }
    }
    for(auto it = cs_union_mems.begin(); it != cs_union_mems.end(); it++) {
        if(it->second.first == serverPtr) {
            return it->first;
        }
    }
    return nullptr;
}

void *getServerDevPtr(void *ptr) {
    if(server_dev_mems.find(ptr) != server_dev_mems.end()) {
        return ptr;
    }
    return nullptr;
}

void freeDevPtr(void *ptr) {
    if(cs_union_mems.find(ptr) != cs_union_mems.end()) {
        free(ptr);
        cs_union_mems.erase(ptr);
    } else if(server_dev_mems.find(ptr) != server_dev_mems.end()) {
        server_dev_mems.erase(ptr);
    }
}

void updateTmpPtr(void *clientPtr, void *serverPtr) {
    if(clientPtr == nullptr || serverPtr == nullptr) {
        return;
    }
    auto it1 = cs_host_tmp_mems.find(clientPtr);
    if(it1 != cs_host_tmp_mems.end()) {
        it1->second.first = serverPtr;
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
                    // 检查是否是数组
                    if(*name_end == '[') {
                        param->is_array = 1;
                        param->array_size = atoi(name_end + 1);
                    }
                    *name_end = '\0';
                    strncpy(param->name, name_start, sizeof(param->name) - 1);
                } else {
                    strncpy(param->name, name_start, sizeof(param->name) - 1);
                }
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
extern "C" void mem2server(RpcConn *conn, void **serverPtr, void *clientPtr, ssize_t size) {
#ifdef DEBUG
    std::cout << "Hook: mem2server called " << clientPtr << " " << size << std::endl;
#endif
    void *ptr = nullptr; // 服务器端内存指针(起始位置)
    size_t memSize = 0;

    if(clientPtr == nullptr) {
        *serverPtr = nullptr;
        return;
    }
    // 纯设备指针，不用同步内存数据
    auto it = server_dev_mems.find(clientPtr);
    if(it != server_dev_mems.end()) {
        *serverPtr = (void *)it->first;
        return;
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
    if(ptr == nullptr) {
        auto it = cs_host_tmp_mems.find(clientPtr);
        if(it != cs_host_tmp_mems.end()) {
            ptr = it->second.first;
            *serverPtr = ptr;
            memSize = it->second.second;
        }
    }
    // ptr为空表示clientPtr是未知的主机内存
    if(ptr == nullptr) {
        if(size == 0) { // 大小不知道，无法同步数据
            printf("WARNING: no size info for conn host memory 0x%p\n", clientPtr);
            *serverPtr = clientPtr;
            return;
        }
        // 写入null指针
        void **tmp_ptr = (void **)conn->get_iov_buffer(sizeof(ptr));
        if(tmp_ptr == nullptr) {
            printf("WARNING: failed to get iov buffer for conn host memory 0x%p\n", clientPtr);
            *serverPtr = clientPtr;
            return;
        }
        *tmp_ptr = ptr;
        conn->write(tmp_ptr, sizeof(*tmp_ptr));

        // 写入客户端内存数据
        conn->write(clientPtr, size, true);

        // 读取服务器端内存指针
        conn->read(serverPtr, sizeof(*serverPtr));

        // 用于记录客户端内存地址和服务器端创建的临时内存地址之间的映射
        cs_host_tmp_mems[clientPtr] = std::make_pair(nullptr, size);
    } else {
        // 写入服务器端内存指针
        void **tmp_ptr = (void **)conn->get_iov_buffer(sizeof(ptr));
        if(tmp_ptr == nullptr) {
            printf("WARNING: failed to get iov buffer for conn host memory 0x%p\n", clientPtr);
            *serverPtr = clientPtr;
            return;
        }
        *tmp_ptr = ptr;
        conn->write(tmp_ptr, sizeof(*tmp_ptr));

        // 写入客户端内存数据
        conn->write(clientPtr, memSize, true);
    }
    return;
}

// 准备从服务器向客户端同步内存
extern "C" void mem2client(RpcConn *conn, void *clientPtr, ssize_t size, bool del_tmp_ptr) {
#ifdef DEBUG
    std::cout << "Hook: mem2client called " << clientPtr << " " << size << std::endl;
#endif
    void *ptr = nullptr; // 服务器端内存指针(起始位置)
    size_t memSize = 0;

    if(clientPtr == nullptr) {
        return;
    }
    // 纯设备指针，不用同步内存数据
    auto it = server_dev_mems.find(clientPtr);
    if(it != server_dev_mems.end()) {
        return;
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
                if(it->second.first == nullptr) {
                    continue;
                }
                if((uintptr_t)clientPtr >= (uintptr_t)it->first && (uintptr_t)clientPtr < ((uintptr_t)it->first + it->second.second)) {
                    clientPtr = (void *)(uintptr_t)it->first; // 修改clientPtr到内存的起始位置
                    ptr = (void *)(uintptr_t)it->second.first;
                    memSize = it->second.second;
                    break;
                }
            }
        }
    }
    // ptr为空表示clientPtr是未知的主机内存, 查看是否在服务器端已经创建对应的临时内存
    if(ptr == nullptr) {
        auto it = cs_host_tmp_mems.find(clientPtr);
        if(it != cs_host_tmp_mems.end()) {
            ptr = it->second.first;
            memSize = it->second.second;
            if(del_tmp_ptr) {
                cs_host_tmp_mems.erase(clientPtr);
            }
        }
    }
    // ptr为空表示clientPtr是未知的主机内存
    if(ptr == nullptr) {
        printf("WARNING: unknown server side host memory for conn pointer: %p\n", clientPtr);
        return;
    }
    if(memSize > 0) {
        // 写入服务器端内存指针
        void **tmp_ptr = (void **)conn->get_iov_buffer(sizeof(ptr));
        if(tmp_ptr == nullptr) {
            printf("WARNING: failed to get iov buffer for conn host memory 0x%p\n", clientPtr);
            return;
        }
        *tmp_ptr = ptr;
        conn->write(tmp_ptr, sizeof(*tmp_ptr));

        // 写入是否删除临时内存
        int *int_ptr = (int *)conn->get_iov_buffer(sizeof(int));
        if(int_ptr == nullptr) {
            printf("WARNING: failed to get iov buffer for conn host memory 0x%p\n", clientPtr);
            return;
        }
        *int_ptr = del_tmp_ptr;
        conn->write(int_ptr, sizeof(*int_ptr));

        // 写入大小
        ssize_t *tmp_size = (ssize_t *)conn->get_iov_buffer(sizeof(size));
        if(tmp_size == nullptr) {
            printf("WARNING: failed to get iov buffer for conn host memory 0x%p\n", clientPtr);
            return;
        }
        *tmp_size = memSize;
        conn->write(tmp_size, sizeof(*tmp_size));

        // 读取数据到客户端内存
        conn->read(clientPtr, memSize, true);
    }
    return;
}

extern "C" void mem2client_async(RpcConn *conn, void *clientPtr, ssize_t size, bool del_tmp_ptr) {
#ifdef DEBUG
    std::cout << "Hook: mem2client_async called " << clientPtr << " " << size << std::endl;
#endif
    void *ptr = nullptr; // 服务器端内存指针(起始位置)
    size_t memSize = 0;

    if(clientPtr == nullptr) {
        return;
    }
    // 纯设备指针，不用同步内存数据
    auto it = server_dev_mems.find(clientPtr);
    if(it != server_dev_mems.end()) {
        return;
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
                if(it->second.first == nullptr) {
                    continue;
                }
                if((uintptr_t)clientPtr >= (uintptr_t)it->first && (uintptr_t)clientPtr < ((uintptr_t)it->first + it->second.second)) {
                    clientPtr = (void *)(uintptr_t)it->first; // 修改clientPtr到内存的起始位置
                    ptr = (void *)(uintptr_t)it->second.first;
                    memSize = it->second.second;
                    break;
                }
            }
        }
    }
    // ptr为空表示clientPtr是未知的主机内存, 查看是否在服务器端已经创建对应的临时内存
    if(ptr == nullptr) {
        auto it = cs_host_tmp_mems.find(clientPtr);
        if(it != cs_host_tmp_mems.end()) {
            ptr = it->second.first;
            memSize = it->second.second;
            if(del_tmp_ptr) {
                cs_host_tmp_mems.erase(clientPtr);
            }
        }
    }
    // ptr为空表示clientPtr是未知的主机内存
    if(ptr == nullptr) {
        printf("WARNING: unknown server side host memory for conn pointer: %p\n", clientPtr);
        return;
    }
    if(memSize > 0) {
        // 写入服务器端内存指针
        void **tmp_ptr = (void **)conn->get_iov_buffer(sizeof(ptr));
        if(tmp_ptr == nullptr) {
            printf("WARNING: failed to get iov buffer for conn host memory 0x%p\n", clientPtr);
            return;
        }
        *tmp_ptr = ptr;
        conn->write(tmp_ptr, sizeof(*tmp_ptr));

        // 写入是否删除临时内存
        int *int_ptr = (int *)conn->get_iov_buffer(sizeof(int));
        if(int_ptr == nullptr) {
            printf("WARNING: failed to get iov buffer for conn host memory 0x%p\n", clientPtr);
            return;
        }
        *int_ptr = del_tmp_ptr;
        conn->write(int_ptr, sizeof(*int_ptr));

        // 写入大小
        ssize_t *tmp_size = (ssize_t *)conn->get_iov_buffer(sizeof(size));
        if(tmp_size == nullptr) {
            printf("WARNING: failed to get iov buffer for conn host memory 0x%p\n", clientPtr);
            return;
        }
        *tmp_size = memSize;
        conn->write(tmp_size, sizeof(*tmp_size));

        // 写入客户端指针
        void **tmp_client_ptr = (void **)conn->get_iov_buffer(sizeof(clientPtr));
        if(tmp_client_ptr == nullptr) {
            printf("WARNING: failed to get iov buffer for conn host memory 0x%p\n", clientPtr);
            return;
        }
        *tmp_client_ptr = clientPtr;
        conn->write(tmp_client_ptr, sizeof(*tmp_client_ptr));
    }
    return;
}

extern "C" cudaError_t cudaFree(void *devPtr) {
#ifdef DEBUG
    std::cout << "Hook: cudaFree called" << std::endl;
#endif

    cudaError_t _result;
    void *serverPtr = getServerHostPtr(devPtr);
    if(serverPtr == nullptr) {
        serverPtr = getServerDevPtr(devPtr);
        if(serverPtr == nullptr) {
            serverPtr = devPtr;
        }
    }
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cudaFree);
    conn->write(&serverPtr, sizeof(serverPtr));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == cudaSuccess) {
        freeDevPtr(devPtr);
    }
    rpc_release_conn(conn);
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
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cudaFreeHost);
    conn->write(&serverPtr, sizeof(serverPtr));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == cudaSuccess) {
        free(ptr);
        cs_host_mems.erase(ptr);
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" const char *cudaGetErrorName(cudaError_t error) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetErrorName called" << std::endl;
#endif

    static char _cudaGetErrorName_result[1024];
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cudaGetErrorName);
    conn->write(&error, sizeof(error));
    conn->read(_cudaGetErrorName_result, 1024, true);
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    rpc_release_conn(conn);
    return _cudaGetErrorName_result;
}

extern "C" const char *cudaGetErrorString(cudaError_t error) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetErrorString called" << std::endl;
#endif

    static char _cudaGetErrorString_result[1024];
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cudaGetErrorString);
    conn->write(&error, sizeof(error));
    conn->read(_cudaGetErrorString_result, 1024, true);
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    rpc_release_conn(conn);
    return _cudaGetErrorString_result;
}

extern "C" cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
#ifdef DEBUG
    std::cout << "Hook: cudaGetSymbolAddress called" << std::endl;
#endif

    cudaError_t _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cudaGetSymbolAddress);
    conn->read(devPtr, sizeof(*devPtr));
    conn->write(&symbol, sizeof(symbol));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == cudaSuccess) {
        server_dev_mems[*devPtr] = 0;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, struct cudaMemcpy3DParms *pNodeParams) {
#ifdef DEBUG
    std::cout << "Hook: cudaGraphMemcpyNodeGetParams called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    void *end_flag = (void *)0xffffffff;
    cudaError_t _result;
    conn->prepare_request(RPC_cudaGraphMemcpyNodeGetParams);
    conn->write(&node, sizeof(node));
    conn->read(pNodeParams, sizeof(*pNodeParams));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    void *ptr = getClientHostPtr(pNodeParams->srcPtr.ptr);
    if(ptr != nullptr) {
        pNodeParams->srcPtr.ptr = ptr;
        mem2client(conn, (void *)pNodeParams->srcPtr.ptr, sizeof(pNodeParams->srcPtr.pitch * pNodeParams->srcPtr.ysize), false);
    }
    ptr = getClientHostPtr(pNodeParams->dstPtr.ptr);
    if(ptr != nullptr) {
        pNodeParams->dstPtr.ptr = ptr;
        mem2client(conn, (void *)pNodeParams->dstPtr.ptr, sizeof(pNodeParams->dstPtr.pitch * pNodeParams->dstPtr.ysize), false);
    }
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaHostAlloc called" << std::endl;
#endif

    cudaError_t _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *pHost = malloc(size);
    if(*pHost == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    conn->prepare_request(RPC_cudaHostAlloc);
    conn->read(&serverPtr, sizeof(serverPtr));
    conn->write(&size, sizeof(size));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        free(*pHost);
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == cudaSuccess) {
        cs_host_mems[*pHost] = std::make_pair(serverPtr, size);
    } else {
        free(*pHost);
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaHostRegister called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    void *serverPtr = getServerHostPtr(ptr);
    cudaError_t _result;
    conn->prepare_request(RPC_cudaHostRegister);
    conn->write(&serverPtr, sizeof(serverPtr));
    conn->write(&size, sizeof(size));
    conn->write(&flags, sizeof(flags));
    conn->write(ptr, size);
    conn->read(&serverPtr, sizeof(serverPtr));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == cudaSuccess) {
        cs_host_mems[ptr] = std::make_pair(serverPtr, size);
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaHostUnregister(void *ptr) {
#ifdef DEBUG
    std::cout << "Hook: cudaHostUnregister called" << std::endl;
#endif

    void *serverPtr = getServerHostPtr(ptr);
    if(serverPtr == nullptr) {
        return cudaErrorUnknown;
    }
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    cudaError_t _result;
    conn->prepare_request(RPC_cudaHostUnregister);
    conn->write(&serverPtr, sizeof(serverPtr));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    cs_host_mems.erase(ptr);
    rpc_release_conn(conn);
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
        return cudaErrorInvalidDeviceFunction;
    }
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    for(int i = 0; i < f->param_count; i++) {
        mem2server(conn, &f->params[i].ptr, *((void **)args[i]), -1);
    }
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }

    conn->prepare_request(RPC_cudaLaunchKernel);
    conn->write(&func, sizeof(func));
    conn->write(&gridDim, sizeof(gridDim));
    conn->write(&blockDim, sizeof(blockDim));
    conn->write(&sharedMem, sizeof(sharedMem));
    conn->write(&stream, sizeof(stream));
    conn->write(&f->param_count, sizeof(f->param_count));

    for(int i = 0; i < f->param_count; i++) {
        conn->write(&f->params[i].ptr, f->params[i].size, true);
        updateTmpPtr(*((void **)args[i]), f->params[i].ptr);
    }
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    // TODO 由于cudaLaunchKernel函数是异步调用的，所以此时还得不到返回值，
    // 需要实现一个对应的回调机制，让服务器端在执行完cudaLaunchKernel函数后，将返回值写回给客户端
    // 每个客户端需要启动一个线程，用于接收服务器端的返回值
    // 服务器端需要知道的信息：
    // 1. 一个内存区域的服务器端指针和客户端指针，还有大小
    // 所以这里需要让服务器启动一个回调任务.
    conn->prepare_request(RPC_mem2client_async);
    for(int i = 0; i < f->param_count; i++) {
        mem2client_async(conn, *((void **)args[i]), -1, false);
    }
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        conn->write(&stream, sizeof(stream));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
#ifdef DEBUG
    std::cout << "Hook: cudaLaunchCooperativeKernel called" << std::endl;
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
        return cudaErrorInvalidDeviceFunction;
    }
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    for(int i = 0; i < f->param_count; i++) {
        mem2server(conn, &f->params[i].ptr, *((void **)args[i]), -1);
    }
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }

    conn->prepare_request(RPC_cudaLaunchCooperativeKernel);
    conn->write(&func, sizeof(func));
    conn->write(&gridDim, sizeof(gridDim));
    conn->write(&blockDim, sizeof(blockDim));
    conn->write(&sharedMem, sizeof(sharedMem));
    conn->write(&stream, sizeof(stream));
    conn->write(&f->param_count, sizeof(f->param_count));

    for(int i = 0; i < f->param_count; i++) {
        conn->write(&f->params[i].ptr, f->params[i].size, true);
        updateTmpPtr(*((void **)args[i]), f->params[i].ptr);
    }
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    for(int i = 0; i < f->param_count; i++) {
        mem2client(conn, *((void **)args[i]), -1, false);
    }
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
#ifdef DEBUG
    std::cout << "Hook: cudaMalloc called" << std::endl;
#endif

    cudaError_t _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cudaMalloc);
    conn->read(devPtr, sizeof(*devPtr));
    conn->write(&size, sizeof(size));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == cudaSuccess) {
        server_dev_mems[*devPtr] = size;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr, struct cudaExtent extent) {
#ifdef DEBUG
    std::cout << "Hook: cudaMalloc3D called" << std::endl;
#endif

    cudaError_t _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cudaMalloc3D);
    conn->read(pitchedDevPtr, sizeof(*pitchedDevPtr));
    conn->write(&extent, sizeof(extent));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == cudaSuccess) {
        server_dev_mems[pitchedDevPtr->ptr] = pitchedDevPtr->pitch * extent.height;
    }
    rpc_release_conn(conn);
    return _result;
}

// 是不是只在客户端这边做客户端主机内存和服务器主机内存的映射就够了？
extern "C" cudaError_t cudaMallocHost(void **ptr, size_t size) {
#ifdef DEBUG
    std::cout << "Hook: cudaMallocHost called" << std::endl;
#endif

    cudaError_t _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *ptr = malloc(size);
    if(*ptr == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    conn->prepare_request(RPC_cudaMallocHost);
    conn->read(&serverPtr, sizeof(serverPtr));
    conn->write(&size, sizeof(size));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        free(*ptr);
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == cudaSuccess) {
        cs_host_mems[*ptr] = std::make_pair(serverPtr, size);
    } else {
        free(*ptr);
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cudaMallocManaged called" << std::endl;
#endif

    cudaError_t _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *devPtr = malloc(size);
    if(*devPtr == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    conn->prepare_request(RPC_cudaMallocManaged);
    conn->read(&serverPtr, sizeof(serverPtr));
    conn->write(&size, sizeof(size));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        free(*devPtr);
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == cudaSuccess) {
        cs_union_mems[*devPtr] = std::make_pair((void *)serverPtr, size);
    } else {
        free(*devPtr);
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) {
#ifdef DEBUG
    std::cout << "Hook: cudaMallocPitch called" << std::endl;
#endif

    cudaError_t _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cudaMallocPitch);
    conn->read(devPtr, sizeof(*devPtr));
    conn->read(pitch, sizeof(*pitch));
    conn->write(&width, sizeof(width));
    conn->write(&height, sizeof(height));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == cudaSuccess) {
        server_dev_mems[*devPtr] = *pitch * height;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t cudaMemRangeGetAttributes(void **data, size_t *dataSizes, enum cudaMemRangeAttribute *attributes, size_t numAttributes, const void *devPtr, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cudaMemRangeGetAttributes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }

    void *_0devPtr;
    mem2server(conn, &_0devPtr, (void *)devPtr, -1);
    cudaError_t _result;
    conn->prepare_request(RPC_cudaMemRangeGetAttributes);
    conn->write(&numAttributes, sizeof(numAttributes));
    conn->write(&_0devPtr, sizeof(_0devPtr));
    conn->write(&count, sizeof(count));
    conn->write(dataSizes, sizeof(*dataSizes) * numAttributes);
    conn->write(attributes, sizeof(*attributes) * numAttributes);
    for(size_t i = 0; i < numAttributes; i++) {
        conn->read(data[i], dataSizes[i]);
    }
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }

    rpc_release_conn(conn);
    return _result;
}

extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream) {
#ifdef DEBUG
    std::cout << "Hook: __cudaPopCallConfiguration called" << std::endl;
#endif

    cudaError_t _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC___cudaPopCallConfiguration);
    conn->read(gridDim, sizeof(*gridDim));
    conn->read(blockDim, sizeof(*blockDim));
    conn->read(sharedMem, sizeof(*sharedMem));
    conn->read(stream, sizeof(cudaStream_t));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, struct CUstream_st *stream) {
#ifdef DEBUG
    std::cout << "Hook: __cudaPushCallConfiguration called" << std::endl;
#endif

    unsigned _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC___cudaPushCallConfiguration);
    conn->write(&gridDim, sizeof(gridDim));
    conn->write(&blockDim, sizeof(blockDim));
    conn->write(&sharedMem, sizeof(sharedMem));
    conn->write(&stream, sizeof(stream));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    rpc_release_conn(conn);
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

    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC___cudaRegisterFatBinary);
    conn->write(binary, sizeof(__cudaFatCudaBinary2));
    conn->write(header, size, true);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
#ifdef DEBUG
    std::cout << "Hook: __cudaRegisterFatBinaryEnd called" << std::endl;
#endif

    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC___cudaRegisterFatBinaryEnd);
    conn->write(&fatCubinHandle, sizeof(fatCubinHandle));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    rpc_release_conn(conn);
}

extern "C" void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
#ifdef DEBUG
    std::cout << "Hook: __cudaRegisterFunction called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC___cudaRegisterFunction);
    conn->write(&fatCubinHandle, sizeof(fatCubinHandle));
    conn->write(&hostFun, sizeof(hostFun));
    conn->write(deviceName, strlen(deviceName) + 1, true);
    conn->write(&thread_limit, sizeof(thread_limit));
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
    conn->write(&mask, sizeof(mask));
    if(tid != nullptr) {
        conn->write(tid, sizeof(uint3));
    }
    if(bid != nullptr) {
        conn->write(bid, sizeof(uint3));
    }
    if(bDim != nullptr) {
        conn->write(bDim, sizeof(dim3));
    }
    if(gDim != nullptr) {
        conn->write(gDim, sizeof(dim3));
    }
    if(wSize != nullptr) {
        conn->write(wSize, sizeof(int));
    }
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    // also memorize the host pointer function
    for(auto &funcinfo : funcinfos) {
        if(strcmp(funcinfo.name, deviceName) == 0) {
            funcinfo.fun_ptr = (void *)hostFun;
            printf("================ funcinfo.fun_ptr: %p\n", funcinfo.fun_ptr);
        }
    }
    rpc_release_conn(conn);
}

extern "C" void __cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress, char *deviceAddress, const char *deviceName, int ext, size_t size, int constant, int global) {
#ifdef DEBUG
    std::cout << "Hook: __cudaRegisterManagedVar called" << std::endl;
#endif

    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC___cudaRegisterManagedVar);
    conn->write(&fatCubinHandle, sizeof(fatCubinHandle));
    conn->write(deviceName, strlen(deviceName) + 1, true);
    conn->write(&ext, sizeof(ext));
    conn->write(&size, sizeof(size));
    conn->write(&constant, sizeof(constant));
    conn->write(&global, sizeof(global));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }

    rpc_release_conn(conn);
}

extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, size_t size, int constant, int global) {
#ifdef DEBUG
    std::cout << "Hook: __cudaRegisterVar called" << std::endl;
#endif

    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC___cudaRegisterVar);
    conn->write(&fatCubinHandle, sizeof(fatCubinHandle));
    conn->write(&hostVar, sizeof(hostVar));
    conn->write(deviceName, strlen(deviceName) + 1, true);
    conn->write(&ext, sizeof(ext));
    conn->write(&size, sizeof(size));
    conn->write(&constant, sizeof(constant));
    conn->write(&global, sizeof(global));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    server_dev_mems[(void *)hostVar] = size;
    rpc_release_conn(conn);
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle) {
#ifdef DEBUG
    std::cout << "Hook: __cudaUnregisterFatBinary called" << std::endl;
#endif

    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC___cudaUnregisterFatBinary);
    conn->write(&fatCubinHandle, sizeof(fatCubinHandle));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    rpc_release_conn(conn);
}

extern "C" char __cudaInitModule(void **fatCubinHandle) {
#ifdef DEBUG
    std::cout << "Hook: __cudaInitModule called" << std::endl;
#endif

    char _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC___cudaInitModule);
    // PARAM void **fatCubinHandle
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    // PARAM void **fatCubinHandle
    rpc_release_conn(conn);
    return _result;
}

// CUDA Driver API (cu*)
extern "C" CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuExternalMemoryGetMappedBuffer called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuExternalMemoryGetMappedBuffer);
    conn->read(devPtr, sizeof(*devPtr));
    conn->write(&extMem, sizeof(extMem));
    conn->write(bufferDesc, sizeof(*bufferDesc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*devPtr] = bufferDesc->size;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGetErrorName(CUresult error, const char **pStr) {
#ifdef DEBUG
    std::cout << "Hook: cuGetErrorName called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuGetErrorName);
    conn->write(&error, sizeof(error));
    static char _cuGetErrorName_pStr[1024];
    conn->read(_cuGetErrorName_pStr, 1024, true);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    *pStr = _cuGetErrorName_pStr;
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGetErrorString(CUresult error, const char **pStr) {
#ifdef DEBUG
    std::cout << "Hook: cuGetErrorString called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuGetErrorString);
    conn->write(&error, sizeof(error));
    static char _cuGetErrorString_pStr[1024];
    conn->read(_cuGetErrorString_pStr, 1024, true);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    *pStr = _cuGetErrorString_pStr;
    rpc_release_conn(conn);
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
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuGraphicsResourceGetMappedPointer_v2);
    conn->read(pDevPtr, sizeof(*pDevPtr));
    conn->read(pSize, sizeof(*pSize));
    conn->write(&resource, sizeof(resource));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*pDevPtr] = *pSize;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuImportExternalMemory(CUexternalMemory *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc) {
#ifdef DEBUG
    std::cout << "Hook: cuImportExternalMemory called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuImportExternalMemory);
    conn->read(extMem_out, sizeof(*extMem_out));
    conn->write(memHandleDesc, sizeof(*memHandleDesc));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*extMem_out] = memHandleDesc->size;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuIpcOpenMemHandle_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuIpcOpenMemHandle_v2);
    conn->read(pdptr, sizeof(*pdptr));
    conn->write(&handle, sizeof(handle));
    conn->write(&Flags, sizeof(Flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*pdptr] = 0;
    }
    rpc_release_conn(conn);
    return _result;
}

#if CUDA_VERSION > 11040
extern "C" CUresult cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuLibraryGetGlobal called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuLibraryGetGlobal);
    conn->read(dptr, sizeof(*dptr));
    conn->read(bytes, sizeof(*bytes));
    conn->write(&library, sizeof(library));
    conn->write(name, strlen(name) + 1, true);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr] = *bytes;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuLibraryGetManaged called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuLibraryGetManaged);
    conn->read(dptr, sizeof(*dptr));
    conn->read(bytes, sizeof(*bytes));
    conn->write(&library, sizeof(library));
    conn->write(name, strlen(name) + 1, true);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
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
    rpc_release_conn(conn);
    return _result;
}
#endif

extern "C" CUresult cuLaunchCooperativeKernel(CUfunction func, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams) {
#ifdef DEBUG
    std::cout << "Hook: cuLaunchCooperativeKernel called" << std::endl;
#endif

    CUresult _result;
    FuncInfo *f = nullptr;
    for(auto &funcinfo : funcinfos) {
        if(funcinfo.fun_ptr == func) {
            f = &funcinfo;
            break;
        }
    }
    if(f == nullptr) {
        return CUDA_ERROR_LAUNCH_FAILED;
    }
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_mem2server);
    for(int i = 0; i < f->param_count; i++) {
        mem2server(conn, &f->params[i].ptr, *((void **)kernelParams[i]), -1);
    }
    void *end_flag = (void *)0xffffffff;
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }

    conn->prepare_request(RPC_cuLaunchCooperativeKernel);
    conn->write(&func, sizeof(func));
    conn->write(&gridDimX, sizeof(gridDimX));
    conn->write(&gridDimY, sizeof(gridDimY));
    conn->write(&gridDimZ, sizeof(gridDimZ));
    conn->write(&blockDimX, sizeof(blockDimX));
    conn->write(&blockDimY, sizeof(blockDimY));
    conn->write(&blockDimZ, sizeof(blockDimZ));
    conn->write(&sharedMemBytes, sizeof(sharedMemBytes));
    conn->write(&hStream, sizeof(hStream));
    conn->write(&f->param_count, sizeof(f->param_count));

    for(int i = 0; i < f->param_count; i++) {
        conn->write(&f->params[i].ptr, f->params[i].size, true);
        updateTmpPtr(*((void **)kernelParams[i]), f->params[i].ptr);
    }
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    conn->prepare_request(RPC_mem2client);
    for(int i = 0; i < f->param_count; i++) {
        mem2client(conn, *((void **)kernelParams[i]), -1, false);
    }
    if(conn->get_iov_send_count(true) > 0) {
        conn->write(&end_flag, sizeof(end_flag));
        if(conn->submit_request() != RpcError::OK) {
            std::cerr << "Failed to submit request" << std::endl;
            rpc_release_conn(conn);
            exit(1);
        }
    }
    rpc_release_conn(conn);
    return _result;
}

// 此函数用于保留一段连续的虚拟地址空间，供后续的内存映射操作使用
extern "C" CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAddressReserve called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuMemAddressReserve);
    conn->read(ptr, sizeof(*ptr));
    conn->write(&size, sizeof(size));
    conn->write(&alignment, sizeof(alignment));
    conn->write(&addr, sizeof(addr));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        munmap((void *)*ptr, size);
        rpc_release_conn(conn);
        exit(1);
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAlloc_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuMemAlloc_v2);
    conn->read(dptr, sizeof(*dptr));
    conn->write(&bytesize, sizeof(bytesize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr] = bytesize;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAllocHost_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *pp = malloc(bytesize);
    if(*pp == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    conn->prepare_request(RPC_cuMemAllocHost_v2);
    conn->read(&serverPtr, sizeof(serverPtr));
    conn->write(&bytesize, sizeof(bytesize));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        free(*pp);
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        cs_host_mems[*pp] = std::make_pair(serverPtr, bytesize);
    } else {
        free(*pp);
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAllocManaged called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *dptr = (CUdeviceptr)malloc(bytesize);
    if(*dptr == 0) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    conn->prepare_request(RPC_cuMemAllocManaged);
    conn->read(dptr, sizeof(*dptr));
    conn->write(&bytesize, sizeof(bytesize));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        cs_union_mems[(void *)*dptr] = std::make_pair((void *)serverPtr, bytesize);
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
#ifdef DEBUG
    std::cout << "Hook: cuMemAllocPitch_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuMemAllocPitch_v2);
    conn->read(dptr, sizeof(*dptr));
    conn->read(pPitch, sizeof(*pPitch));
    conn->write(&WidthInBytes, sizeof(WidthInBytes));
    conn->write(&Height, sizeof(Height));
    conn->write(&ElementSizeBytes, sizeof(ElementSizeBytes));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr] = *pPitch * Height;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemCreate called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuMemCreate);
    conn->read(handle, sizeof(*handle));
    conn->write(&size, sizeof(size));
    conn->write(prop, sizeof(*prop));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
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
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemFreeHost(void *p) {
#ifdef DEBUG
    std::cout << "Hook: cuMemFreeHost called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    void *serverPtr;
    if(cs_host_mems.find(p) == cs_host_mems.end()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    serverPtr = cs_host_mems[p].first;
    conn->prepare_request(RPC_cuMemFreeHost);
    conn->write(&serverPtr, sizeof(serverPtr));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        free(p);
        cs_host_mems.erase(p);
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
#ifdef DEBUG
    std::cout << "Hook: cuMemGetAddressRange_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuMemGetAddressRange_v2);
    conn->read(pbase, sizeof(*pbase));
    conn->read(psize, sizeof(*psize));
    conn->write(&dptr, sizeof(dptr));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*pbase] = *psize;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemRangeGetAttributes(void **data, size_t *dataSizes, CUmem_range_attribute *attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) {
#ifdef DEBUG
    std::cout << "Hook: cuMemRangeGetAttributes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }

    void *_0devPtr;
    mem2server(conn, &_0devPtr, (void *)devPtr, -1);
    CUresult _result;
    conn->prepare_request(RPC_cuMemRangeGetAttributes);
    conn->write(&numAttributes, sizeof(numAttributes));
    conn->write(&_0devPtr, sizeof(_0devPtr));
    conn->write(&count, sizeof(count));
    conn->write(dataSizes, sizeof(*dataSizes) * numAttributes);
    conn->write(attributes, sizeof(*attributes) * numAttributes);
    for(size_t i = 0; i < numAttributes; i++) {
        conn->read(data[i], dataSizes[i]);
    }
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }

    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemHostAlloc called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    void *serverPtr;
    *pp = malloc(bytesize);
    if(*pp == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    conn->prepare_request(RPC_cuMemHostAlloc);
    conn->read(&serverPtr, sizeof(serverPtr));
    conn->write(&bytesize, sizeof(bytesize));
    conn->write(&Flags, sizeof(Flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        free(*pp);
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        cs_host_mems[*pp] = std::make_pair(serverPtr, bytesize);
    } else {
        free(*pp);
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemHostGetDevicePointer_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuMemHostGetDevicePointer_v2);
    conn->read(pdptr, sizeof(*pdptr));
    void *serverPtr = getServerHostPtr(p);
    if(serverPtr == nullptr) {
        conn->write(&p, sizeof(p));
    } else {
        conn->write(&serverPtr, sizeof(serverPtr));
    }
    conn->write(&Flags, sizeof(Flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*pdptr] = 0;
    }
    rpc_release_conn(conn);
    return _result;
}

// 此函数将物理内存块映射到预先保留的虚拟地址空间，使该内存可被GPU或CPU访问
extern "C" CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) {
#ifdef DEBUG
    std::cout << "Hook: cuMemMap called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }

    conn->prepare_request(RPC_cuMemMap);
    conn->write(&ptr, sizeof(ptr));
    conn->write(&size, sizeof(size));
    conn->write(&offset, sizeof(offset));
    conn->write(&handle, sizeof(handle));
    conn->write(&flags, sizeof(flags));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }

    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData *shareData) {
#ifdef DEBUG
    std::cout << "Hook: cuMemPoolImportPointer called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuMemPoolImportPointer);
    conn->read(ptr_out, sizeof(*ptr_out));
    conn->write(&pool, sizeof(pool));
    conn->write(shareData, sizeof(*shareData));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*ptr_out] = 0;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
#ifdef DEBUG
    std::cout << "Hook: cuMemRelease called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuMemRelease);
    conn->write(&handle, sizeof(handle));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        host_handles.erase(handle);
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
#ifdef DEBUG
    std::cout << "Hook: cuModuleGetGlobal_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuModuleGetGlobal_v2);
    conn->read(dptr, sizeof(*dptr));
    conn->read(bytes, sizeof(*bytes));
    conn->write(&hmod, sizeof(hmod));
    conn->write(name, strlen(name) + 1, true);
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr] = *bytes;
    }
    rpc_release_conn(conn);
    return _result;
}

static size_t getAttributeSize(CUpointer_attribute attribute) {
    switch(attribute) {
    // 4-byte attributes
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
    case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
    case CU_POINTER_ATTRIBUTE_IS_MANAGED:
    case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
    case CU_POINTER_ATTRIBUTE_MAPPED:
    case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
    case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
    case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS:
        return sizeof(unsigned int); // 4 bytes

    // 8-byte attributes
    case CU_POINTER_ATTRIBUTE_CONTEXT:
        return sizeof(CUcontext); // 8 bytes (pointer-sized)
    case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
        return sizeof(CUdeviceptr); // 8 bytes
    case CU_POINTER_ATTRIBUTE_HOST_POINTER:
        return sizeof(void *); // 8 bytes
    case CU_POINTER_ATTRIBUTE_BUFFER_ID:
        return sizeof(unsigned long long); // 8 bytes
    case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
        return sizeof(void *); // 8 bytes
    case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
        return sizeof(size_t); // 8 bytes
    case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
        return sizeof(CUmemoryPool); // 8 bytes

    // 4-byte (int)
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
        return sizeof(int); // 4 bytes

    // 16-byte (P2P tokens)
    case CU_POINTER_ATTRIBUTE_P2P_TOKENS:
        return 2 * sizeof(unsigned long long); // 16 bytes

    default:
        // Unknown attribute, return 0 or handle error
        return 0;
    }
}

extern "C" CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr) {
#ifdef DEBUG
    std::cout << "Hook: cuPointerGetAttributes called" << std::endl;
#endif
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    CUresult _result;
    conn->prepare_request(RPC_cuPointerGetAttributes);
    conn->write(&numAttributes, sizeof(numAttributes));
    conn->write(&ptr, sizeof(ptr));
    conn->write(attributes, sizeof(*attributes) * numAttributes);
    for(int i = 0; i < numAttributes; i++) {
        conn->read(data[i], getAttributeSize(attributes[i]), false);
    }
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuTexRefGetAddress_v2(CUdeviceptr *pdptr, CUtexref hTexRef) {
#ifdef DEBUG
    std::cout << "Hook: cuTexRefGetAddress_v2 called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuTexRefGetAddress_v2);
    conn->read(pdptr, sizeof(*pdptr));
    conn->write(&hTexRef, sizeof(hTexRef));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*pdptr] = 0;
    }
    rpc_release_conn(conn);
    return _result;
}

extern "C" CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr *dptr_out) {
#ifdef DEBUG
    std::cout << "Hook: cuGraphMemFreeNodeGetParams called" << std::endl;
#endif

    CUresult _result;
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_cuGraphMemFreeNodeGetParams);
    conn->write(&hNode, sizeof(hNode));
    conn->read(dptr_out, sizeof(*dptr_out));
    conn->read(&_result, sizeof(_result));
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    if(_result == CUDA_SUCCESS) {
        server_dev_mems[(void *)*dptr_out] = 0; // TODO?
    }
    rpc_release_conn(conn);
    return _result;
}

// NVML (nvml*)

extern "C" const char *nvmlErrorString(nvmlReturn_t result) {
#ifdef DEBUG
    std::cout << "Hook: nvmlErrorString called" << std::endl;
#endif

    static char _nvmlErrorString_result[1024];
    RpcConn *conn = rpc_get_conn();
    if(conn == nullptr) {
        std::cerr << "Failed to get rpc conn" << std::endl;
        exit(1);
    }
    conn->prepare_request(RPC_nvmlErrorString);
    conn->write(&result, sizeof(result));
    conn->read(_nvmlErrorString_result, 1024, true);
    if(conn->submit_request() != RpcError::OK) {
        std::cerr << "Failed to submit request" << std::endl;
        rpc_release_conn(conn);
        exit(1);
    }
    rpc_release_conn(conn);
    return _nvmlErrorString_result;
}
