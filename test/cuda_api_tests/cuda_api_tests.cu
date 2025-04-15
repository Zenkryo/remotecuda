#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdexcept>

// 定义设备端的全局变量
__device__ int dev_data;

// 简单的核函数，用于测试
__global__ void test_kernel() {
    // 空核函数，仅用于测试启动
}

class CudaApiTest : public ::testing::Test {
  protected:
    CUdevice device;
    CUcontext context;

    void SetUp() override {
        CUresult result = cuInit(0);
        if(result != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to initialize CUDA");
        }

        result = cuDeviceGet(&device, 0);
        if(result != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to get CUDA device");
        }

        result = cuCtxCreate(&context, 0, device);
        if(result != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to create CUDA context");
        }
    }

    void TearDown() override {
        if(context) {
            cuCtxDestroy(context);
        }
    }
};

// 辅助函数用于检查CUDA错误
void checkCudaError(cudaError_t error, const char *message, const char *file, int line) {
    if(error != cudaSuccess) {
        const char *errorName = cudaGetErrorName(error);
        const char *errorString = cudaGetErrorString(error);
        FAIL() << "Error at " << file << ":" << line << " - " << message << ": " << errorName << " - " << errorString;
    }
}

void checkCuError(CUresult result, const char *message, const char *file, int line) {
    if(result != CUDA_SUCCESS) {
        const char *errorName;
        cuGetErrorName(result, &errorName);
        const char *errorString;
        cuGetErrorString(result, &errorString);
        FAIL() << "Error at " << file << ":" << line << " - " << message << ": " << errorName << " - " << errorString;
    }
}

// 宏定义用于简化错误检查调用
#define CHECK_CUDA_ERROR(err, msg) checkCudaError(err, msg, __FILE__, __LINE__)
#define CHECK_CU_ERROR(result, msg) checkCuError(result, msg, __FILE__, __LINE__)

// Test cudaFree
TEST_F(CudaApiTest, CudaFree) {
    void *devPtr;
    cudaError_t err = cudaMalloc(&devPtr, 1024);
    CHECK_CUDA_ERROR(err, "Failed to allocate device memory");
    ASSERT_NE(devPtr, nullptr);

    err = cudaFree(devPtr);
    CHECK_CUDA_ERROR(err, "Failed to free device memory");
}

// Test cudaFreeHost
TEST_F(CudaApiTest, CudaFreeHost) {
    void *hostPtr;
    cudaError_t err = cudaMallocHost(&hostPtr, 1024);
    CHECK_CUDA_ERROR(err, "Failed to allocate host memory");
    ASSERT_NE(hostPtr, nullptr);

    err = cudaFreeHost(hostPtr);
    CHECK_CUDA_ERROR(err, "Failed to free host memory");
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
    cudaError_t err = cudaGetSymbolAddress(&devPtr, dev_data);
    CHECK_CUDA_ERROR(err, "Failed to get symbol address");
    ASSERT_NE(devPtr, nullptr);

    // Write a value to the device variable
    int testValue = 42;
    err = cudaMemcpy(devPtr, &testValue, sizeof(int), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err, "Failed to write to device variable");

    // Read back the value
    int readValue = 0;
    err = cudaMemcpy(&readValue, devPtr, sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err, "Failed to read from device variable");

    // Verify the value
    ASSERT_EQ(readValue, testValue);

    // Test with a different value
    testValue = 123;
    err = cudaMemcpy(devPtr, &testValue, sizeof(int), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err, "Failed to write second value to device variable");

    err = cudaMemcpy(&readValue, devPtr, sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err, "Failed to read second value from device variable");

    ASSERT_EQ(readValue, testValue);
}

// Test cudaHostAlloc with different flags
TEST_F(CudaApiTest, CudaHostAlloc) {
    struct TestCase {
        unsigned int flags;
        const char *description;
    };

    TestCase testCases[] = {{cudaHostAllocDefault, "Default flags"}, {cudaHostAllocPortable, "Portable memory"}, {cudaHostAllocMapped, "Mapped memory"}, {cudaHostAllocWriteCombined, "Write-combined memory"}};

    for(const auto &testCase : testCases) {
        void *hostPtr;
        cudaError_t err = cudaHostAlloc(&hostPtr, 1024, testCase.flags);
        CHECK_CUDA_ERROR(err, (std::string("Failed to allocate host memory with ") + testCase.description).c_str());
        ASSERT_NE(hostPtr, nullptr);

        // Test writing to host memory
        int *intPtr = static_cast<int *>(hostPtr);
        for(int i = 0; i < 256; i++) { // 1024 bytes / sizeof(int) = 256 ints
            intPtr[i] = i;
        }

        // Test reading from host memory
        for(int i = 0; i < 256; i++) {
            ASSERT_EQ(intPtr[i], i) << "Memory verification failed at index " << i << " with flags " << testCase.description;
        }

        // Test writing a different pattern
        for(int i = 0; i < 256; i++) {
            intPtr[i] = 255 - i;
        }

        // Verify the new pattern
        for(int i = 0; i < 256; i++) {
            ASSERT_EQ(intPtr[i], 255 - i) << "Memory verification failed at index " << i << " with flags " << testCase.description;
        }

        err = cudaFreeHost(hostPtr);
        CHECK_CUDA_ERROR(err, (std::string("Failed to free host memory with ") + testCase.description).c_str());
    }
}

// Test cudaHostRegister and cudaHostUnregister with different flags
TEST_F(CudaApiTest, CudaHostRegister) {
    struct TestCase {
        unsigned int flags;
        const char *description;
        bool optional; // 标记是否为可选功能
    };

    TestCase testCases[] = {
        {cudaHostRegisterDefault, "Default flags", false}, {cudaHostRegisterPortable, "Portable memory", false}, {cudaHostRegisterMapped, "Mapped memory", false}, {cudaHostRegisterIoMemory, "I/O memory", true} // 标记为可选功能
    };

    for(const auto &testCase : testCases) {
        void *hostPtr = malloc(1024);
        ASSERT_NE(hostPtr, nullptr);

        cudaError_t err = cudaHostRegister(hostPtr, 1024, testCase.flags);

        if(testCase.optional) {
            // 对于可选功能，如果返回不支持的错误，我们认为是正常的
            if(err == cudaErrorInvalidValue || err == cudaErrorNotSupported) {
                // 跳过这个测试用例
                free(hostPtr);
                continue;
            }
        }

        CHECK_CUDA_ERROR(err, (std::string("Failed to register host memory with ") + testCase.description).c_str());

        err = cudaHostUnregister(hostPtr);
        CHECK_CUDA_ERROR(err, (std::string("Failed to unregister host memory with ") + testCase.description).c_str());

        free(hostPtr);
    }
}

// Test cudaLaunchKernel with different grid and block dimensions
TEST_F(CudaApiTest, CudaLaunchKernel) {
    struct TestCase {
        dim3 gridDim;
        dim3 blockDim;
        const char *description;
    };

    TestCase testCases[] = {{dim3(1, 1, 1), dim3(1, 1, 1), "1x1x1 grid and block"}, {dim3(2, 2, 1), dim3(32, 32, 1), "2x2x1 grid and 32x32x1 block"}, {dim3(4, 4, 1), dim3(16, 16, 1), "4x4x1 grid and 16x16x1 block"}};

    for(const auto &testCase : testCases) {
        void *args[] = {};
        cudaError_t err = cudaLaunchKernel((const void *)test_kernel, testCase.gridDim, testCase.blockDim, args, 0, 0);
        CHECK_CUDA_ERROR(err, (std::string("Failed to launch kernel with ") + testCase.description).c_str());

        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err, (std::string("Failed to synchronize after kernel launch with ") + testCase.description).c_str());
    }
}

// Test cudaMalloc with different sizes
TEST_F(CudaApiTest, CudaMalloc) {
    size_t sizes[] = {1, 1024, 1024 * 1024}; // Remove 1GB test

    for(size_t size : sizes) {
        void *devPtr;
        cudaError_t err = cudaMalloc(&devPtr, size);
        CHECK_CUDA_ERROR(err, "Failed to allocate device memory");
        ASSERT_NE(devPtr, nullptr);

        // Allocate host memory for testing
        void *hostPtr = malloc(size);
        ASSERT_NE(hostPtr, nullptr);

        // Initialize host memory with a pattern
        int *hostIntPtr = static_cast<int *>(hostPtr);
        size_t numInts = size / sizeof(int);
        for(size_t i = 0; i < numInts; i++) {
            hostIntPtr[i] = static_cast<int>(i);
        }

        // Copy from host to device
        err = cudaMemcpy(devPtr, hostPtr, size, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err, "Failed to copy from host to device");

        // Clear host memory
        memset(hostPtr, 0, size);

        // Copy back from device to host
        err = cudaMemcpy(hostPtr, devPtr, size, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(err, "Failed to copy from device to host");

        // Verify the data
        for(size_t i = 0; i < numInts; i++) {
            ASSERT_EQ(hostIntPtr[i], static_cast<int>(i)) << "Memory verification failed at index " << i << " with size " << size;
        }

        // Test with a different pattern
        for(size_t i = 0; i < numInts; i++) {
            hostIntPtr[i] = static_cast<int>(numInts - 1 - i);
        }

        // Copy the new pattern to device
        err = cudaMemcpy(devPtr, hostPtr, size, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err, "Failed to copy second pattern to device");

        // Clear host memory again
        memset(hostPtr, 0, size);

        // Copy back from device to host
        err = cudaMemcpy(hostPtr, devPtr, size, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(err, "Failed to copy second pattern from device");

        // Verify the new pattern
        for(size_t i = 0; i < numInts; i++) {
            ASSERT_EQ(hostIntPtr[i], static_cast<int>(numInts - 1 - i)) << "Second pattern verification failed at index " << i << " with size " << size;
        }

        // Clean up
        free(hostPtr);
        err = cudaFree(devPtr);
        CHECK_CUDA_ERROR(err, "Failed to free device memory");
    }
}

// Test cudaMalloc3D with different extents
TEST_F(CudaApiTest, CudaMalloc3D) {
    struct TestCase {
        cudaExtent extent;
        const char *description;
    };

    TestCase testCases[] = {{make_cudaExtent(32, 32, 1), "32x32x1"}, {make_cudaExtent(64, 64, 1), "64x64x1"}, {make_cudaExtent(128, 128, 1), "128x128x1"}};

    for(const auto &testCase : testCases) {
        cudaPitchedPtr pitchedDevPtr;
        cudaError_t err = cudaMalloc3D(&pitchedDevPtr, testCase.extent);
        CHECK_CUDA_ERROR(err, (std::string("Failed to allocate 3D memory with extent ") + testCase.description).c_str());
        ASSERT_NE(pitchedDevPtr.ptr, nullptr);

        // // Create host memory for testing
        // size_t hostPitch = testCase.extent.width * sizeof(int);
        // size_t hostSize = hostPitch * testCase.extent.height * testCase.extent.depth;
        // void *hostPtr = malloc(hostSize);
        // ASSERT_NE(hostPtr, nullptr);

        // // Initialize host memory with a pattern
        // int *hostIntPtr = static_cast<int *>(hostPtr);
        // for(size_t z = 0; z < testCase.extent.depth; z++) {
        //     for(size_t y = 0; y < testCase.extent.height; y++) {
        //         for(size_t x = 0; x < testCase.extent.width; x++) {
        //             size_t index = z * testCase.extent.height * testCase.extent.width + y * testCase.extent.width + x;
        //             hostIntPtr[index] = static_cast<int>(index);
        //         }
        //     }
        // }

        // // Copy from host to device
        // cudaMemcpy3DParms copyParams = {0};
        // copyParams.srcPtr = make_cudaPitchedPtr(hostPtr, hostPitch, testCase.extent.width, testCase.extent.height);
        // copyParams.dstPtr = pitchedDevPtr;
        // copyParams.extent = testCase.extent;
        // copyParams.kind = cudaMemcpyHostToDevice;

        // err = cudaMemcpy3D(&copyParams);
        // CHECK_CUDA_ERROR(err, (std::string("Failed to copy from host to device with extent ") + testCase.description).c_str());

        // // Clear host memory
        // memset(hostPtr, 0, hostSize);

        // // Copy back from device to host
        // copyParams.srcPtr = pitchedDevPtr;
        // copyParams.dstPtr = make_cudaPitchedPtr(hostPtr, hostPitch, testCase.extent.width, testCase.extent.height);
        // copyParams.kind = cudaMemcpyDeviceToHost;

        // err = cudaMemcpy3D(&copyParams);
        // CHECK_CUDA_ERROR(err, (std::string("Failed to copy from device to host with extent ") + testCase.description).c_str());

        // // Verify the data
        // for(size_t z = 0; z < testCase.extent.depth; z++) {
        //     for(size_t y = 0; y < testCase.extent.height; y++) {
        //         for(size_t x = 0; x < testCase.extent.width; x++) {
        //             size_t index = z * testCase.extent.height * testCase.extent.width + y * testCase.extent.width + x;
        //             ASSERT_EQ(hostIntPtr[index], static_cast<int>(index)) << "Memory verification failed at position (" << x << "," << y << "," << z << ") with extent " << testCase.description;
        //         }
        //     }
        // }

        // // Test with a different pattern
        // for(size_t z = 0; z < testCase.extent.depth; z++) {
        //     for(size_t y = 0; y < testCase.extent.height; y++) {
        //         for(size_t x = 0; x < testCase.extent.width; x++) {
        //             size_t index = z * testCase.extent.height * testCase.extent.width + y * testCase.extent.width + x;
        //             hostIntPtr[index] = static_cast<int>(testCase.extent.width * testCase.extent.height * testCase.extent.depth - 1 - index);
        //         }
        //     }
        // }

        // // Copy the new pattern to device
        // copyParams.srcPtr = make_cudaPitchedPtr(hostPtr, hostPitch, testCase.extent.width, testCase.extent.height);
        // copyParams.dstPtr = pitchedDevPtr;
        // copyParams.kind = cudaMemcpyHostToDevice;

        // err = cudaMemcpy3D(&copyParams);
        // CHECK_CUDA_ERROR(err, (std::string("Failed to copy second pattern to device with extent ") + testCase.description).c_str());

        // // Clear host memory again
        // memset(hostPtr, 0, hostSize);

        // // Copy back from device to host
        // copyParams.srcPtr = pitchedDevPtr;
        // copyParams.dstPtr = make_cudaPitchedPtr(hostPtr, hostPitch, testCase.extent.width, testCase.extent.height);
        // copyParams.kind = cudaMemcpyDeviceToHost;

        // err = cudaMemcpy3D(&copyParams);
        // CHECK_CUDA_ERROR(err, (std::string("Failed to copy second pattern from device with extent ") + testCase.description).c_str());

        // // Verify the new pattern
        // for(size_t z = 0; z < testCase.extent.depth; z++) {
        //     for(size_t y = 0; y < testCase.extent.height; y++) {
        //         for(size_t x = 0; x < testCase.extent.width; x++) {
        //             size_t index = z * testCase.extent.height * testCase.extent.width + y * testCase.extent.width + x;
        //             ASSERT_EQ(hostIntPtr[index], static_cast<int>(testCase.extent.width * testCase.extent.height * testCase.extent.depth - 1 - index)) << "Second pattern verification failed at position (" << x << "," << y << "," << z << ") with extent " << testCase.description;
        //         }
        //     }
        // }

        // // Clean up
        // free(hostPtr);
        err = cudaFree(pitchedDevPtr.ptr);
        CHECK_CUDA_ERROR(err, (std::string("Failed to free 3D memory with extent ") + testCase.description).c_str());
    }
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

    // Test writing to managed memory
    int *intPtr = static_cast<int *>(devPtr);
    for(int i = 0; i < 256; i++) { // 1024 bytes / sizeof(int) = 256 ints
        intPtr[i] = i;
    }

    // Test reading from managed memory
    for(int i = 0; i < 256; i++) {
        ASSERT_EQ(intPtr[i], i) << "Managed memory verification failed at index " << i;
    }

    // Test with a different pattern
    for(int i = 0; i < 256; i++) {
        intPtr[i] = 255 - i;
    }

    // Verify the new pattern
    for(int i = 0; i < 256; i++) {
        ASSERT_EQ(intPtr[i], 255 - i) << "Second pattern verification failed at index " << i;
    }

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

    // Allocate host memory with the same pitch
    void *hostPtr = malloc(pitch * 32);
    ASSERT_NE(hostPtr, nullptr);

    // Initialize host memory with a pattern
    int *hostIntPtr = static_cast<int *>(hostPtr);
    for(int y = 0; y < 32; y++) {
        for(int x = 0; x < 32; x++) {
            hostIntPtr[y * (pitch / sizeof(int)) + x] = y * 32 + x;
        }
    }

    // Copy from host to device
    err = cudaMemcpy2D(devPtr, pitch, hostPtr, pitch, 32 * sizeof(int), 32, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);

    // Clear host memory
    memset(hostPtr, 0, pitch * 32);

    // Copy back from device to host
    err = cudaMemcpy2D(hostPtr, pitch, devPtr, pitch, 32 * sizeof(int), 32, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);

    // Verify the data
    for(int y = 0; y < 32; y++) {
        for(int x = 0; x < 32; x++) {
            ASSERT_EQ(hostIntPtr[y * (pitch / sizeof(int)) + x], y * 32 + x) << "Memory verification failed at position (" << x << "," << y << ")";
        }
    }

    // Test with a different pattern
    for(int y = 0; y < 32; y++) {
        for(int x = 0; x < 32; x++) {
            hostIntPtr[y * (pitch / sizeof(int)) + x] = (31 - y) * 32 + (31 - x);
        }
    }

    // Copy the new pattern to device
    err = cudaMemcpy2D(devPtr, pitch, hostPtr, pitch, 32 * sizeof(int), 32, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);

    // Clear host memory again
    memset(hostPtr, 0, pitch * 32);

    // Copy back from device to host
    err = cudaMemcpy2D(hostPtr, pitch, devPtr, pitch, 32 * sizeof(int), 32, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);

    // Verify the new pattern
    for(int y = 0; y < 32; y++) {
        for(int x = 0; x < 32; x++) {
            ASSERT_EQ(hostIntPtr[y * (pitch / sizeof(int)) + x], (31 - y) * 32 + (31 - x)) << "Second pattern verification failed at position (" << x << "," << y << ")";
        }
    }

    // Clean up
    free(hostPtr);
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

    // Allocate host memory for testing
    void *hostPtr = malloc(1024);
    ASSERT_NE(hostPtr, nullptr);

    // Initialize host memory with a pattern
    int *hostIntPtr = static_cast<int *>(hostPtr);
    for(int i = 0; i < 256; i++) { // 1024 bytes / sizeof(int) = 256 ints
        hostIntPtr[i] = i;
    }

    // Copy from host to device
    result = cuMemcpyHtoD_v2(devPtr, hostPtr, 1024);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Clear host memory
    memset(hostPtr, 0, 1024);

    // Copy back from device to host
    result = cuMemcpyDtoH_v2(hostPtr, devPtr, 1024);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Verify the data
    for(int i = 0; i < 256; i++) {
        ASSERT_EQ(hostIntPtr[i], i) << "Memory verification failed at index " << i;
    }

    // Test with a different pattern
    for(int i = 0; i < 256; i++) {
        hostIntPtr[i] = 255 - i;
    }

    // Copy the new pattern to device
    result = cuMemcpyHtoD_v2(devPtr, hostPtr, 1024);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Clear host memory again
    memset(hostPtr, 0, 1024);

    // Copy back from device to host
    result = cuMemcpyDtoH_v2(hostPtr, devPtr, 1024);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Verify the new pattern
    for(int i = 0; i < 256; i++) {
        ASSERT_EQ(hostIntPtr[i], 255 - i) << "Second pattern verification failed at index " << i;
    }

    // Test cuMemFree_v2
    result = cuMemFree_v2(devPtr);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Test cuMemAllocHost_v2
    void *hostPtr2;
    result = cuMemAllocHost_v2(&hostPtr2, 1024);
    ASSERT_EQ(result, CUDA_SUCCESS);
    ASSERT_NE(hostPtr2, nullptr);

    // Test writing to host memory
    int *hostIntPtr2 = static_cast<int *>(hostPtr2);
    for(int i = 0; i < 256; i++) {
        hostIntPtr2[i] = i;
    }

    // Test reading from host memory
    for(int i = 0; i < 256; i++) {
        ASSERT_EQ(hostIntPtr2[i], i) << "Host memory verification failed at index " << i;
    }

    // Test with a different pattern
    for(int i = 0; i < 256; i++) {
        hostIntPtr2[i] = 255 - i;
    }

    // Verify the new pattern
    for(int i = 0; i < 256; i++) {
        ASSERT_EQ(hostIntPtr2[i], 255 - i) << "Second host pattern verification failed at index " << i;
    }

    // Test cuMemFreeHost
    result = cuMemFreeHost(hostPtr2);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Clean up
    free(hostPtr);
}

// Test cuExternalMemoryGetMappedBuffer
TEST_F(CudaApiTest, CuExternalMemoryGetMappedBuffer) {
    CUresult result;
    CUdeviceptr devPtr = 0;
    CUexternalMemory extMem = nullptr;
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
    CUgraphicsResource resource = nullptr;

    result = cuGraphicsResourceGetMappedPointer_v2(&devPtr, &size, resource);
    // This test is expected to fail since we don't have a valid graphics resource
    ASSERT_NE(result, CUDA_SUCCESS);
    ASSERT_EQ(result, CUDA_ERROR_INVALID_HANDLE);
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
    CUdeviceptr devPtr = 0;
    CUipcMemHandle handle = {};
    unsigned int flags = 0;

    result = cuIpcOpenMemHandle_v2(&devPtr, handle, flags);
    // This test is expected to fail since we don't have a valid IPC memory handle
    ASSERT_NE(result, CUDA_SUCCESS);
}

// Test cuLaunchCooperativeKernel
TEST_F(CudaApiTest, CuLaunchCooperativeKernel) {
    CUresult result;
    CUfunction func = nullptr;
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

    // 设置虚拟内存分配参数
    size_t allocationSize = 1 << 24; // 16MB
    size_t alignment = 1 << 16;      // 64KB对齐

    // 保留虚拟地址范围
    result = cuMemAddressReserve(&ptr, allocationSize, alignment, 0, 0);

    ASSERT_EQ(result, CUDA_SUCCESS);
    ASSERT_NE(ptr, 0);

    // Clean up
    result = cuMemAddressFree(ptr, allocationSize);
    ASSERT_EQ(result, CUDA_SUCCESS);
}

// Test cuMemCreate
TEST_F(CudaApiTest, CuMemCreate) {
    CUresult result;

    // 查询设备属性
    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;

    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // 确保大小是粒度的整数倍
    size_t size = ((1024 * 1024) + granularity - 1) & ~(granularity - 1); // 1MB对齐

    // 创建内存分配
    CUmemGenericAllocationHandle handle;
    result = cuMemCreate(&handle, size, &prop, 0);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // 保留虚拟地址范围
    CUdeviceptr ptr;
    result = cuMemAddressReserve(&ptr, size, granularity, 0, 0);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // 映射内存
    result = cuMemMap(ptr, size, 0, handle, 0);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // 设置访问权限
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    result = cuMemSetAccess(ptr, size, &accessDesc, 1);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Allocate host memory for testing
    void *hostPtr = malloc(size);
    ASSERT_NE(hostPtr, nullptr);

    // Initialize host memory with a pattern
    int *hostIntPtr = static_cast<int *>(hostPtr);
    size_t numInts = size / sizeof(int);
    for(size_t i = 0; i < numInts; i++) {
        hostIntPtr[i] = static_cast<int>(i);
    }

    // Copy from host to device
    result = cuMemcpyHtoD_v2(ptr, hostPtr, size);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Clear host memory
    memset(hostPtr, 0, size);

    // Copy back from device to host
    result = cuMemcpyDtoH_v2(hostPtr, ptr, size);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Verify the data
    for(size_t i = 0; i < numInts; i++) {
        ASSERT_EQ(hostIntPtr[i], static_cast<int>(i)) << "Memory verification failed at index " << i;
    }

    // Test with a different pattern
    for(size_t i = 0; i < numInts; i++) {
        hostIntPtr[i] = static_cast<int>(numInts - 1 - i);
    }

    // Copy the new pattern to device
    result = cuMemcpyHtoD_v2(ptr, hostPtr, size);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Clear host memory again
    memset(hostPtr, 0, size);

    // Copy back from device to host
    result = cuMemcpyDtoH_v2(hostPtr, ptr, size);
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Verify the new pattern
    for(size_t i = 0; i < numInts; i++) {
        ASSERT_EQ(hostIntPtr[i], static_cast<int>(numInts - 1 - i)) << "Second pattern verification failed at index " << i;
    }

    // Clean up
    result = cuMemUnmap(ptr, size);
    ASSERT_EQ(result, CUDA_SUCCESS);

    result = cuMemAddressFree(ptr, size);
    ASSERT_EQ(result, CUDA_SUCCESS);

    result = cuMemRelease(handle);
    ASSERT_EQ(result, CUDA_SUCCESS);

    free(hostPtr);
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
    CUdeviceptr ptr = 0;
    unsigned int flags = 0;
    size_t offset = 0;

    // 查询设备属性
    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;

    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // 确保大小是粒度的整数倍
    size_t size = ((1024 * 1024) + granularity - 1) & ~(granularity - 1); // 1MB对齐

    // 创建内存分配
    CUmemGenericAllocationHandle handle;
    result = cuMemCreate(&handle, size, &prop, 0);
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
    CUdeviceptr ptr_out = 0;
    CUmemoryPool pool = nullptr;
    CUmemPoolPtrExportData shareData = {};

    result = cuMemPoolImportPointer(&ptr_out, pool, &shareData);
    // This test is expected to fail since we don't have a valid memory pool
    ASSERT_NE(result, CUDA_SUCCESS);
}

// Test cuMemRelease
TEST_F(CudaApiTest, CuMemRelease) {
    CUresult result;

    // 查询设备属性
    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;

    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // 确保大小是粒度的整数倍
    size_t size = ((1024 * 1024) + granularity - 1) & ~(granularity - 1); // 1MB对齐

    // 创建内存分配
    CUmemGenericAllocationHandle handle;
    result = cuMemCreate(&handle, size, &prop, 0);
    ASSERT_EQ(result, CUDA_SUCCESS);

    result = cuMemRelease(handle);
    ASSERT_EQ(result, CUDA_SUCCESS);
}

// Test cuModuleGetGlobal_v2
TEST_F(CudaApiTest, CuModuleGetGlobal) {
    CUresult result;
    CUdeviceptr devPtr = 0;
    size_t bytes = 0;
    CUmodule hmod = nullptr;
    const char *name = "test";

    result = cuModuleGetGlobal_v2(&devPtr, &bytes, hmod, name);
    // This test is expected to fail since we don't have a valid module handle
    ASSERT_NE(result, CUDA_SUCCESS);
}

// Test cuPointerGetAttributes
TEST_F(CudaApiTest, CuPointerGetAttributes) {
    CUresult result;

    // 分配设备内存
    size_t size = 1024 * 1024; // 1MB
    CUdeviceptr d_ptr;
    cuMemAlloc(&d_ptr, size);

    // 准备查询指针属性
    CUpointer_attribute attributes[3] = {CU_POINTER_ATTRIBUTE_CONTEXT, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL};

    // 查询指针属性
    void *data[3];
    CUcontext ctx;
    unsigned int mem_type;
    int device_ordinal;

    data[0] = &ctx;
    data[1] = &mem_type;
    data[2] = &device_ordinal;

    result = cuPointerGetAttributes(3,          // 属性数量
                                    attributes, // 属性数组
                                    data,       // 结果数据数组
                                    d_ptr       // 要查询的指针
    );
    ASSERT_EQ(result, CUDA_SUCCESS);

    // Clean up
    result = cuMemFree_v2(d_ptr);
    ASSERT_EQ(result, CUDA_SUCCESS);
}

// Test cuTexRefGetAddress
TEST_F(CudaApiTest, CuTexRefGetAddress) {
    CUresult result;
    CUdeviceptr devPtr = 0;
    CUtexref hTexRef = nullptr;

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

    // Test 1: Invalid node handle
    {
        CUgraphNode hNode = nullptr;
        CUdeviceptr dptr_out = 0;
        result = cuGraphMemFreeNodeGetParams(hNode, &dptr_out);
        ASSERT_NE(result, CUDA_SUCCESS);
    }

    // Test 2: Create a valid graph and memory free node
    {
        // Create a graph
        CUgraph graph;
        result = cuGraphCreate(&graph, 0);
        ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to create graph";

        // Allocate device memory
        CUdeviceptr dptr;
        result = cuMemAlloc_v2(&dptr, 1024);
        ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to allocate device memory";

        // Create a memory free node
        CUgraphNode hNode;
        CUgraphNode dependencies[] = {}; // Empty dependencies array
        result = cuGraphAddMemFreeNode(&hNode, graph, dependencies, 0, dptr);
        if(result != CUDA_SUCCESS) {
            const char *errorName;
            cuGetErrorName(result, &errorName);
            const char *errorString;
            cuGetErrorString(result, &errorString);
            FAIL() << "Failed to create memory free node: " << errorName << " - " << errorString;
        }

        // Get node parameters
        CUdeviceptr dptr_out = 0;
        result = cuGraphMemFreeNodeGetParams(hNode, &dptr_out);
        ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to get node parameters";
        ASSERT_EQ(dptr_out, dptr) << "Device pointer mismatch";

        // Clean up
        result = cuGraphDestroy(graph);
        ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to destroy graph";
    }

    // Test 3: Create multiple memory free nodes in a graph
    {
        // Create a graph
        CUgraph graph;
        result = cuGraphCreate(&graph, 0);
        ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to create graph";

        // Allocate multiple device memory blocks
        const int numBlocks = 3;
        CUdeviceptr dptrs[numBlocks];
        CUgraphNode nodes[numBlocks];
        CUgraphNode dependencies[] = {}; // Empty dependencies array

        for(int i = 0; i < numBlocks; i++) {
            // Allocate device memory
            result = cuMemAlloc_v2(&dptrs[i], 1024);
            ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to allocate device memory block " << i;

            // Create a memory free node
            result = cuGraphAddMemFreeNode(&nodes[i], graph, dependencies, 0, dptrs[i]);
            if(result != CUDA_SUCCESS) {
                const char *errorName;
                cuGetErrorName(result, &errorName);
                const char *errorString;
                cuGetErrorString(result, &errorString);
                FAIL() << "Failed to create memory free node " << i << ": " << errorName << " - " << errorString;
            }

            // Get node parameters
            CUdeviceptr dptr_out = 0;
            result = cuGraphMemFreeNodeGetParams(nodes[i], &dptr_out);
            ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to get node parameters for node " << i;
            ASSERT_EQ(dptr_out, dptrs[i]) << "Device pointer mismatch for node " << i;
        }

        // Clean up
        result = cuGraphDestroy(graph);
        ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to destroy graph";

        // Free device memory
        for(int i = 0; i < numBlocks; i++) {
            result = cuMemFree_v2(dptrs[i]);
            ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to free device memory block " << i;
        }
    }

    // Test 4: Create a graph with dependencies between memory free nodes
    {
        // Create a graph
        CUgraph graph;
        result = cuGraphCreate(&graph, 0);
        ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to create graph";

        // Allocate device memory
        CUdeviceptr dptr1, dptr2;
        result = cuMemAlloc_v2(&dptr1, 1024);
        ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to allocate first device memory block";
        result = cuMemAlloc_v2(&dptr2, 1024);
        ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to allocate second device memory block";

        // Create memory free nodes
        CUgraphNode hNode1, hNode2;
        CUgraphNode dependencies1[] = {}; // Empty dependencies array for first node
        result = cuGraphAddMemFreeNode(&hNode1, graph, dependencies1, 0, dptr1);
        if(result != CUDA_SUCCESS) {
            const char *errorName;
            cuGetErrorName(result, &errorName);
            const char *errorString;
            cuGetErrorString(result, &errorString);
            FAIL() << "Failed to create first memory free node: " << errorName << " - " << errorString;
        }

        CUgraphNode dependencies2[] = {hNode1}; // Second node depends on first node
        result = cuGraphAddMemFreeNode(&hNode2, graph, dependencies2, 1, dptr2);
        if(result != CUDA_SUCCESS) {
            const char *errorName;
            cuGetErrorName(result, &errorName);
            const char *errorString;
            cuGetErrorString(result, &errorString);
            FAIL() << "Failed to create second memory free node: " << errorName << " - " << errorString;
        }

        // Get node parameters
        CUdeviceptr dptr_out1 = 0, dptr_out2 = 0;
        result = cuGraphMemFreeNodeGetParams(hNode1, &dptr_out1);
        ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to get first node parameters";
        ASSERT_EQ(dptr_out1, dptr1) << "First device pointer mismatch";

        result = cuGraphMemFreeNodeGetParams(hNode2, &dptr_out2);
        ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to get second node parameters";
        ASSERT_EQ(dptr_out2, dptr2) << "Second device pointer mismatch";

        // Clean up
        result = cuGraphDestroy(graph);
        ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to destroy graph";

        // Free device memory
        result = cuMemFree_v2(dptr1);
        ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to free first device memory block";
        result = cuMemFree_v2(dptr2);
        ASSERT_EQ(result, CUDA_SUCCESS) << "Failed to free second device memory block";
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
