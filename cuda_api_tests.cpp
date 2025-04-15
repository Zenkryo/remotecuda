#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <cstring>

// Define CUDA kernels at file scope
__global__ void testKernel() {}

class CudaApiTest : public ::testing::Test {
  protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
    }

    void TearDown() override {
        cudaError_t err = cudaDeviceReset();
        ASSERT_EQ(err, cudaSuccess);
    }
};

// Test cudaFree
TEST_F(CudaApiTest, CudaFree) {
    void *devPtr;
    cudaError_t err = cudaMalloc(&devPtr, 1024);
    ASSERT_EQ(err, cudaSuccess);

    err = cudaFree(devPtr);
    ASSERT_EQ(err, cudaSuccess);
}

// Test cudaFreeHost
TEST_F(CudaApiTest, CudaFreeHost) {
    void *hostPtr;
    cudaError_t err = cudaMallocHost(&hostPtr, 1024);
    ASSERT_EQ(err, cudaSuccess);

    err = cudaFreeHost(hostPtr);
    ASSERT_EQ(err, cudaSuccess);
}

// Test cudaGetErrorName and cudaGetErrorString
TEST_F(CudaApiTest, CudaGetErrorInfo) {
    const char *errorName = cudaGetErrorName(cudaSuccess);
    ASSERT_NE(errorName, nullptr);
    ASSERT_STREQ(errorName, "cudaSuccess");

    const char *errorString = cudaGetErrorString(cudaSuccess);
    ASSERT_NE(errorString, nullptr);
    ASSERT_STREQ(errorString, "no error");
}

// Test cudaGetSymbolAddress
TEST_F(CudaApiTest, CudaGetSymbolAddress) {
    void *devPtr;
    cudaError_t err = cudaGetSymbolAddress(&devPtr, (const void *)testKernel);
    ASSERT_EQ(err, cudaSuccess);
    ASSERT_NE(devPtr, nullptr);
}

// Test cudaHostAlloc
TEST_F(CudaApiTest, CudaHostAlloc) {
    void *hostPtr;
    cudaError_t err = cudaHostAlloc(&hostPtr, 1024, cudaHostAllocDefault);
    ASSERT_EQ(err, cudaSuccess);
    ASSERT_NE(hostPtr, nullptr);

    err = cudaFreeHost(hostPtr);
    ASSERT_EQ(err, cudaSuccess);
}

// Test cudaHostRegister and cudaHostUnregister
TEST_F(CudaApiTest, CudaHostRegister) {
    void *hostPtr = malloc(1024);
    ASSERT_NE(hostPtr, nullptr);

    cudaError_t err = cudaHostRegister(hostPtr, 1024, cudaHostRegisterDefault);
    ASSERT_EQ(err, cudaSuccess);

    err = cudaHostUnregister(hostPtr);
    ASSERT_EQ(err, cudaSuccess);

    free(hostPtr);
}

// Test cudaLaunchKernel
TEST_F(CudaApiTest, CudaLaunchKernel) {
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(1, 1, 1);
    void *args[] = {};

    cudaError_t err = cudaLaunchKernel((const void *)testKernel, gridDim, blockDim, args, 0, 0);
    ASSERT_EQ(err, cudaSuccess);

    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess);
}

// Test cudaLaunchCooperativeKernel
TEST_F(CudaApiTest, CudaLaunchCooperativeKernel) {
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(1, 1, 1);
    void *args[] = {};

    cudaError_t err = cudaLaunchCooperativeKernel((const void *)testKernel, gridDim, blockDim, args, 0, 0);
    ASSERT_EQ(err, cudaSuccess);

    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess);
}

// Test cudaMalloc
TEST_F(CudaApiTest, CudaMalloc) {
    void *devPtr;
    cudaError_t err = cudaMalloc(&devPtr, 1024);
    ASSERT_EQ(err, cudaSuccess);
    ASSERT_NE(devPtr, nullptr);

    err = cudaFree(devPtr);
    ASSERT_EQ(err, cudaSuccess);
}

// Test cudaMalloc3D
TEST_F(CudaApiTest, CudaMalloc3D) {
    cudaPitchedPtr pitchedDevPtr;
    cudaExtent extent = make_cudaExtent(32, 32, 1);

    cudaError_t err = cudaMalloc3D(&pitchedDevPtr, extent);
    ASSERT_EQ(err, cudaSuccess);
    ASSERT_NE(pitchedDevPtr.ptr, nullptr);

    err = cudaFree(pitchedDevPtr.ptr);
    ASSERT_EQ(err, cudaSuccess);
}

// Test cudaMallocHost
TEST_F(CudaApiTest, CudaMallocHost) {
    void *hostPtr;
    cudaError_t err = cudaMallocHost(&hostPtr, 1024);
    ASSERT_EQ(err, cudaSuccess);
    ASSERT_NE(hostPtr, nullptr);

    err = cudaFreeHost(hostPtr);
    ASSERT_EQ(err, cudaSuccess);
}

// Test cudaMallocManaged
TEST_F(CudaApiTest, CudaMallocManaged) {
    void *devPtr;
    cudaError_t err = cudaMallocManaged(&devPtr, 1024, cudaMemAttachGlobal);
    ASSERT_EQ(err, cudaSuccess);
    ASSERT_NE(devPtr, nullptr);

    err = cudaFree(devPtr);
    ASSERT_EQ(err, cudaSuccess);
}

// Test cudaMallocPitch
TEST_F(CudaApiTest, CudaMallocPitch) {
    void *devPtr;
    size_t pitch;
    cudaError_t err = cudaMallocPitch(&devPtr, &pitch, 32, 32);
    ASSERT_EQ(err, cudaSuccess);
    ASSERT_NE(devPtr, nullptr);
    ASSERT_GT(pitch, 0);

    err = cudaFree(devPtr);
    ASSERT_EQ(err, cudaSuccess);
}

// Test cudaMemRangeGetAttributes
TEST_F(CudaApiTest, CudaMemRangeGetAttributes) {
    void *devPtr;
    cudaError_t err = cudaMalloc(&devPtr, 1024);
    ASSERT_EQ(err, cudaSuccess);

    cudaMemRangeAttribute attributes[] = {cudaMemRangeAttributeReadMostly};
    size_t dataSizes[] = {sizeof(int)};
    int data[1];
    void *dataPtrs[] = {data};

    err = cudaMemRangeGetAttributes(dataPtrs, dataSizes, attributes, 1, devPtr, 1024);
    ASSERT_EQ(err, cudaSuccess);

    err = cudaFree(devPtr);
    ASSERT_EQ(err, cudaSuccess);
}

// Test CUDA Driver API functions
TEST_F(CudaApiTest, CudaDriverApi) {
    CUresult result;

    // Test cuGetErrorName and cuGetErrorString
    const char *errorName;
    result = cuGetErrorName(CUDA_SUCCESS, &errorName);
    ASSERT_EQ(result, CUDA_SUCCESS);
    ASSERT_NE(errorName, nullptr);

    const char *errorString;
    result = cuGetErrorString(CUDA_SUCCESS, &errorString);
    ASSERT_EQ(result, CUDA_SUCCESS);
    ASSERT_NE(errorString, nullptr);

    // Test cuMemAlloc_v2
    CUdeviceptr devPtr;
    result = cuMemAlloc_v2(&devPtr, 1024);
    ASSERT_EQ(result, CUDA_SUCCESS);
    ASSERT_NE(devPtr, 0);

    // Test cuMemFree_v2
    result = cuMemFree_v2(devPtr);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Test cuMemAllocHost_v2
    void *hostPtr;
    result = cuMemAllocHost_v2(&hostPtr, 1024);
    ASSERT_EQ(result, CUDA_SUCCESS);
    ASSERT_NE(hostPtr, nullptr);

    // Test cuMemFreeHost
    result = cuMemFreeHost(hostPtr);
    ASSERT_EQ(result, CUDA_SUCCESS);
}

// Test cuExternalMemoryGetMappedBuffer
TEST_F(CudaApiTest, CuExternalMemoryGetMappedBuffer) {
    CUresult result;
    CUdeviceptr devPtr;
    CUexternalMemory extMem;
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc = {};

    result = cuExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufferDesc);
    // This test is expected to fail since we don't have a valid external memory handle
    ASSERT_NE(result, CUDA_SUCCESS);
}

// Test cuGraphicsResourceGetMappedPointer_v2
TEST_F(CudaApiTest, CuGraphicsResourceGetMappedPointer) {
    CUresult result;
    CUdeviceptr devPtr;
    size_t size;
    CUgraphicsResource resource;

    result = cuGraphicsResourceGetMappedPointer_v2(&devPtr, &size, resource);
    // This test is expected to fail since we don't have a valid graphics resource
    ASSERT_NE(result, CUDA_SUCCESS);
}

// Test cuImportExternalMemory
TEST_F(CudaApiTest, CuImportExternalMemory) {
    CUresult result;
    CUexternalMemory extMem;
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memHandleDesc = {};

    result = cuImportExternalMemory(&extMem, &memHandleDesc);
    // This test is expected to fail since we don't have a valid memory handle
    ASSERT_NE(result, CUDA_SUCCESS);
}

// Test cuIpcOpenMemHandle_v2
TEST_F(CudaApiTest, CuIpcOpenMemHandle) {
    CUresult result;
    CUdeviceptr devPtr;
    CUipcMemHandle handle;
    unsigned int flags = 0;

    result = cuIpcOpenMemHandle_v2(&devPtr, handle, flags);
    // This test is expected to fail since we don't have a valid IPC memory handle
    ASSERT_NE(result, CUDA_SUCCESS);
}

// Test cuLaunchCooperativeKernel
TEST_F(CudaApiTest, CuLaunchCooperativeKernel) {
    CUresult result;
    CUfunction func;
    unsigned int gridDimX = 1, gridDimY = 1, gridDimZ = 1;
    unsigned int blockDimX = 1, blockDimY = 1, blockDimZ = 1;
    unsigned int sharedMemBytes = 0;
    CUstream hStream = 0;
    void **kernelParams = nullptr;

    result = cuLaunchCooperativeKernel(func, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
    // This test is expected to fail since we don't have a valid function handle
    ASSERT_NE(result, CUDA_SUCCESS);
}

// Test cuMemAddressReserve
TEST_F(CudaApiTest, CuMemAddressReserve) {
    CUresult result;
    CUdeviceptr ptr;
    size_t size = 1024;
    size_t alignment = 4096;
    CUdeviceptr addr = 0;
    unsigned long long flags = 0;

    result = cuMemAddressReserve(&ptr, size, alignment, addr, flags);
    ASSERT_EQ(result, CUDA_SUCCESS);
    ASSERT_NE(ptr, 0);

    // Clean up
    result = cuMemAddressFree(ptr, size);
    ASSERT_EQ(result, CUDA_SUCCESS);
}

// Test cuMemCreate
TEST_F(CudaApiTest, CuMemCreate) {
    CUresult result;
    CUmemGenericAllocationHandle handle;
    size_t size = 1024;
    CUmemAllocationProp prop = {};
    unsigned long long flags = 0;

    result = cuMemCreate(&handle, size, &prop, flags);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Clean up
    result = cuMemRelease(handle);
    ASSERT_EQ(result, CUDA_SUCCESS);
}

// Test cuMemGetAddressRange_v2
TEST_F(CudaApiTest, CuMemGetAddressRange) {
    CUresult result;
    CUdeviceptr base;
    size_t size;
    CUdeviceptr devPtr;

    // First allocate some memory
    result = cuMemAlloc_v2(&devPtr, 1024);
    ASSERT_EQ(result, CUDA_SUCCESS);

    result = cuMemGetAddressRange_v2(&base, &size, devPtr);
    ASSERT_EQ(result, CUDA_SUCCESS);
    ASSERT_NE(base, 0);
    ASSERT_GE(size, 1024);

    // Clean up
    result = cuMemFree_v2(devPtr);
    ASSERT_EQ(result, CUDA_SUCCESS);
}

// Test cuMemRangeGetAttributes
TEST_F(CudaApiTest, CuMemRangeGetAttributes) {
    CUresult result;
    CUdeviceptr devPtr;

    // First allocate some memory
    result = cuMemAlloc_v2(&devPtr, 1024);
    ASSERT_EQ(result, CUDA_SUCCESS);

    CUmem_range_attribute attributes[] = {CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY};
    size_t dataSizes[] = {sizeof(int)};
    int data[1];
    void *dataPtrs[] = {data};

    result = cuMemRangeGetAttributes(dataPtrs, dataSizes, attributes, 1, devPtr, 1024);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Clean up
    result = cuMemFree_v2(devPtr);
    ASSERT_EQ(result, CUDA_SUCCESS);
}

// Test cuMemHostAlloc
TEST_F(CudaApiTest, CuMemHostAlloc) {
    CUresult result;
    void *hostPtr;
    size_t bytesize = 1024;
    unsigned int flags = 0;

    result = cuMemHostAlloc(&hostPtr, bytesize, flags);
    ASSERT_EQ(result, CUDA_SUCCESS);
    ASSERT_NE(hostPtr, nullptr);

    // Clean up
    result = cuMemFreeHost(hostPtr);
    ASSERT_EQ(result, CUDA_SUCCESS);
}

// Test cuMemHostGetDevicePointer_v2
TEST_F(CudaApiTest, CuMemHostGetDevicePointer) {
    CUresult result;
    CUdeviceptr devPtr;
    void *hostPtr;
    unsigned int flags = 0;

    // First allocate host memory
    result = cuMemHostAlloc(&hostPtr, 1024, flags);
    ASSERT_EQ(result, CUDA_SUCCESS);

    result = cuMemHostGetDevicePointer_v2(&devPtr, hostPtr, flags);
    ASSERT_EQ(result, CUDA_SUCCESS);
    ASSERT_NE(devPtr, 0);

    // Clean up
    result = cuMemFreeHost(hostPtr);
    ASSERT_EQ(result, CUDA_SUCCESS);
}

// Test cuMemMap
TEST_F(CudaApiTest, CuMemMap) {
    CUresult result;
    CUdeviceptr ptr;
    size_t size = 1024;
    size_t offset = 0;
    CUmemGenericAllocationHandle handle;
    unsigned long long flags = 0;

    // First create a memory allocation
    CUmemAllocationProp prop = {};
    result = cuMemCreate(&handle, size, &prop, flags);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Reserve an address
    result = cuMemAddressReserve(&ptr, size, 4096, 0, flags);
    ASSERT_EQ(result, CUDA_SUCCESS);

    result = cuMemMap(ptr, size, offset, handle, flags);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Clean up
    result = cuMemUnmap(ptr, size);
    ASSERT_EQ(result, CUDA_SUCCESS);

    result = cuMemAddressFree(ptr, size);
    ASSERT_EQ(result, CUDA_SUCCESS);

    result = cuMemRelease(handle);
    ASSERT_EQ(result, CUDA_SUCCESS);
}

// Test cuMemPoolImportPointer
TEST_F(CudaApiTest, CuMemPoolImportPointer) {
    CUresult result;
    CUdeviceptr ptr_out;
    CUmemoryPool pool;
    CUmemPoolPtrExportData shareData = {};

    result = cuMemPoolImportPointer(&ptr_out, pool, &shareData);
    // This test is expected to fail since we don't have a valid memory pool
    ASSERT_NE(result, CUDA_SUCCESS);
}

// Test cuMemRelease
TEST_F(CudaApiTest, CuMemRelease) {
    CUresult result;
    CUmemGenericAllocationHandle handle;

    // First create a memory allocation
    CUmemAllocationProp prop = {};
    result = cuMemCreate(&handle, 1024, &prop, 0);
    ASSERT_EQ(result, CUDA_SUCCESS);

    result = cuMemRelease(handle);
    ASSERT_EQ(result, CUDA_SUCCESS);
}

// Test cuModuleGetGlobal_v2
TEST_F(CudaApiTest, CuModuleGetGlobal) {
    CUresult result;
    CUdeviceptr devPtr;
    size_t bytes;
    CUmodule hmod;
    const char *name = "test";

    result = cuModuleGetGlobal_v2(&devPtr, &bytes, hmod, name);
    // This test is expected to fail since we don't have a valid module handle
    ASSERT_NE(result, CUDA_SUCCESS);
}

// Test cuPointerGetAttributes
TEST_F(CudaApiTest, CuPointerGetAttributes) {
    CUresult result;
    CUdeviceptr devPtr;

    // First allocate some memory
    result = cuMemAlloc_v2(&devPtr, 1024);
    ASSERT_EQ(result, CUDA_SUCCESS);

    CUpointer_attribute attributes[] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE};
    void *data[1];

    result = cuPointerGetAttributes(1, attributes, data, devPtr);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Clean up
    result = cuMemFree_v2(devPtr);
    ASSERT_EQ(result, CUDA_SUCCESS);
}

// Test cuTexRefGetAddress
TEST_F(CudaApiTest, CuTexRefGetAddress) {
    CUresult result;
    CUdeviceptr devPtr;
    CUtexref hTexRef;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    result = cuTexRefGetAddress(&devPtr, hTexRef);
#pragma GCC diagnostic pop

    // This test is expected to fail since we don't have a valid texture reference
    ASSERT_NE(result, CUDA_SUCCESS);
}

// Test cuGraphMemFreeNodeGetParams
TEST_F(CudaApiTest, CuGraphMemFreeNodeGetParams) {
    CUresult result;
    CUgraphNode hNode;
    CUdeviceptr dptr_out;

    result = cuGraphMemFreeNodeGetParams(hNode, &dptr_out);
    // This test is expected to fail since we don't have a valid graph node
    ASSERT_NE(result, CUDA_SUCCESS);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
