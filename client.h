#ifndef __CLIENT_H__
#define __CLIENT_H__
#include <string.h>
#include "rpc/rpc_core.h"
using namespace rpc;

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

extern std::vector<FuncInfo> funcinfos;
extern void *(*real_dlsym)(void *, const char *);

extern "C" void mem2server(RpcConn *conn, void **serverPtr, void *clientPtr, ssize_t size);
extern "C" void mem2client(RpcConn *conn, void *clientPtr, ssize_t size, bool del_tmp_ptr);
void updateTmpPtr(void *clientPtr, void *serverPtr);
void *get_so_handle(const std::string &so_file);
RpcConn *rpc_get_conn();
void rpc_release_conn(RpcConn *conn);
int sizeofPoolAttribute(int attr);
int sum_group(int *group_size, int group_count);
#endif
