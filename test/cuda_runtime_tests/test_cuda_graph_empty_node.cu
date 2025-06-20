#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGraphEmptyNode){
    cudaGraph_t graph;
    cudaError_t err = cudaGraphCreate(&graph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create graph");
    cudaGraphNode_t emptyNode;
    err = cudaGraphAddEmptyNode(&emptyNode, graph, nullptr, 0);
    CHECK_CUDA_ERROR(err, "Failed to add empty node");
    err = cudaGraphDestroy(graph);
    CHECK_CUDA_ERROR(err, "Failed to destroy graph");
}
