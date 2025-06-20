#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDeviceLimits){
    struct TestCase {
        cudaLimit limit;
        size_t value;
        const char *description;
    };
    TestCase testCases[] = {{cudaLimitStackSize, 4096, "Stack size"}, {cudaLimitPrintfFifoSize, 1048576, "Printf FIFO size"}, {cudaLimitMallocHeapSize, 8 * 1024 * 1024, "Malloc heap size"}};
    for(const auto &testCase : testCases) {
        size_t currentValue;
        cudaError_t err = cudaDeviceGetLimit(&currentValue, testCase.limit);
        CHECK_CUDA_ERROR(err, (std::string("Failed to get ") + testCase.description + " limit").c_str());
        err = cudaDeviceSetLimit(testCase.limit, testCase.value);
        CHECK_CUDA_ERROR(err, (std::string("Failed to set ") + testCase.description + " limit").c_str());
        size_t newValue;
        err = cudaDeviceGetLimit(&newValue, testCase.limit);
        CHECK_CUDA_ERROR(err, (std::string("Failed to verify ") + testCase.description + " limit").c_str());
        ASSERT_GE(newValue, testCase.value) << "Failed to set " << testCase.description << " limit";
        err = cudaDeviceSetLimit(testCase.limit, currentValue);
        CHECK_CUDA_ERROR(err, (std::string("Failed to restore ") + testCase.description + " limit").c_str());
    }
}
