#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGraphMemsetNode){
    cudaGraph_t graph;
    cudaError_t err = cudaGraphCreate(&graph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create graph");
    void *devPtr;
    err = cudaMalloc(&devPtr, 1024);
    CHECK_CUDA_ERROR(err, "Failed to allocate device memory");
    cudaGraphNode_t memsetNode;
    cudaMemsetParams memsetParams = {0};
    memsetParams.dst = devPtr;
    memsetParams.elementSize = 1;
    memsetParams.width = 1024;
    memsetParams.height = 1;
    memsetParams.value = 0x42;
    err = cudaGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams);
    CHECK_CUDA_ERROR(err, "Failed to add memset node");
    cudaMemsetParams retrievedParams;
    err = cudaGraphMemsetNodeGetParams(memsetNode, &retrievedParams);
    CHECK_CUDA_ERROR(err, "Failed to get memset node parameters");
    ASSERT_EQ(retrievedParams.dst, memsetParams.dst) << "Destination pointer mismatch";
    ASSERT_EQ(retrievedParams.value, memsetParams.value) << "Value mismatch";
    memsetParams.value = 0x84;
    err = cudaGraphMemsetNodeSetParams(memsetNode, &memsetParams);
    CHECK_CUDA_ERROR(err, "Failed to set memset node parameters");
    err = cudaGraphDestroy(graph);
    CHECK_CUDA_ERROR(err, "Failed to destroy graph");
    err = cudaFree(devPtr);
    CHECK_CUDA_ERROR(err, "Failed to free device memory");
}
