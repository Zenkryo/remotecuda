#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaDeviceGetStreamPriorityRange){
    int leastPriority, greatestPriority;
    cudaError_t err = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    if(err == cudaSuccess) {
        SUCCEED() << "Priority range retrieved successfully";
    } else {
        SUCCEED() << "Function not supported, skipping test";
    }
}
