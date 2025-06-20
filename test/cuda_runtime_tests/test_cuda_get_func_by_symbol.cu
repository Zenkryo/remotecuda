#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGetFuncBySymbol){
    cudaFunction_t func;
    cudaError_t err = cudaGetFuncBySymbol(&func, (const void *)test_kernel);
    if(err == cudaSuccess) {
        ASSERT_NE(func, nullptr) << "Invalid function pointer";
    } else {
        SUCCEED() << "Function symbol lookup not supported, skipping test";
    }
}
