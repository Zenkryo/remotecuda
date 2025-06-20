#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGetLastError){
    cudaGetLastError();
    cudaError_t peekErr = cudaPeekAtLastError();
    ASSERT_EQ(peekErr, cudaSuccess) << "Unexpected error from cudaPeekAtLastError";
    cudaError_t getErr = cudaGetLastError();
    ASSERT_EQ(getErr, cudaSuccess) << "Unexpected error from cudaGetLastError";
    void *devPtr = nullptr;
    cudaMalloc(&devPtr, (size_t)-1); // This should generate an error
    cudaError_t err = cudaGetLastError();
    ASSERT_NE(err, cudaSuccess) << "Expected error from invalid cudaMalloc";
}
