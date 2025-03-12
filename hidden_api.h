// 隐藏的cuda runtime API
#ifndef HIDDEN_API_H
#define HIDDEN_API_H

#include <cuda_runtime_api.h>
#define __cudaFatMAGIC2 0x466243b1

typedef struct __attribute__((__packed__)) __cudaFatCudaBinaryRec2 {
    uint32_t magic;
    uint32_t version;
    uint64_t text; // points to first text section
    uint64_t data; // points to outside of the file
    uint64_t unknown;
    uint64_t text2; // points to second text section
    uint64_t zero;
} __cudaFatCudaBinary2;

typedef struct __attribute__((__packed__)) __cudaFatCudaBinary2HeaderRec {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint64_t size;
} __cudaFatCudaBinary2Header;

typedef struct __cudaFatCudaBinary2EntryRec {
    unsigned int type;
    unsigned int binary;
    unsigned long long int binarySize;
    unsigned int unknown2;
    unsigned int kindOffset;
    unsigned int unknown3;
    unsigned int unknown4;
    unsigned int name;
    unsigned int nameSize;
    unsigned long long int flags;
    unsigned long long int unknown7;
    unsigned long long int uncompressedBinarySize;
} __cudaFatCudaBinary2Entry;

enum FatBin2EntryType { FATBIN_2_PTX = 0x1 };

#define FATBIN_FLAG_COMPRESS 0x0000000000002000LL

extern "C" char __cudaInitModule(void **fatCubinHandle);

extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, struct CUstream_st *stream = 0);

extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream);

extern "C" void **__cudaRegisterFatBinary(void *fatCubin);

extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle);

extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, size_t size, int constant, int global);

extern "C" void __cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress, char *deviceAddress, const char *deviceName, int ext, size_t size, int constant, int global);

extern "C" void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);

#endif
