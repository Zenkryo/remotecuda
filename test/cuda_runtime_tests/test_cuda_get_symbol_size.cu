#include "common.h"

static __device__ int dev_data1;

TEST_F(CudaRuntimeApiTest, CudaGetSymbolSize) {
    size_t size;
    cudaError_t err = cudaGetSymbolSize(&size, dev_data1);
    CHECK_CUDA_ERROR(err, "Failed to get symbol size");
    ASSERT_EQ(size, sizeof(int)) << "Invalid symbol size";
}
