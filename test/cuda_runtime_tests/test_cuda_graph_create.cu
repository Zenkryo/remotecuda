#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGraphCreate){
    cudaGraph_t graph;
    cudaError_t err = cudaGraphCreate(&graph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create graph");
    ASSERT_NE(graph, nullptr);
    err = cudaGraphDestroy(graph);
    CHECK_CUDA_ERROR(err, "Failed to destroy graph");
}
