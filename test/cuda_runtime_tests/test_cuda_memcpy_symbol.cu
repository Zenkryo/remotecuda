#include "common.h"

__device__ float g_dev_symbol;

TEST_F(CudaRuntimeApiTest, CudaMemcpySymbol) {
    float hostValue = 42.0f;
    float retrievedValue = 0.0f;
    cudaError_t err = cudaMemcpyToSymbol(g_dev_symbol, &hostValue, sizeof(float));
    CHECK_CUDA_ERROR(err, "Failed to copy to symbol");
    err = cudaMemcpyFromSymbol(&retrievedValue, g_dev_symbol, sizeof(float));
    CHECK_CUDA_ERROR(err, "Failed to copy from symbol");
    ASSERT_EQ(retrievedValue, hostValue) << "Symbol copy failed";
}
