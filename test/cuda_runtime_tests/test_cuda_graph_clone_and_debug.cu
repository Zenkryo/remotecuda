#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGraphCloneAndDebug){
    cudaGraph_t graph;
    cudaError_t err = cudaGraphCreate(&graph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create graph");
    cudaGraphNode_t node;
    void *kernelArgs[] = {NULL};
    cudaKernelNodeParams kernelParams = {0};
    kernelParams.func = (void *)test_kernel;
    kernelParams.gridDim = dim3(1, 1, 1);
    kernelParams.blockDim = dim3(1, 1, 1);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = NULL;
    err = cudaGraphAddKernelNode(&node, graph, NULL, 0, &kernelParams);
    CHECK_CUDA_ERROR(err, "Failed to add kernel node");
    cudaGraph_t clonedGraph;
    err = cudaGraphClone(&clonedGraph, graph);
    CHECK_CUDA_ERROR(err, "Failed to clone graph");
    err = cudaGraphDebugDotPrint(graph, "original_graph.dot", 0);
    if(err != cudaErrorNotSupported) {
        CHECK_CUDA_ERROR(err, "Failed to print debug dot file");
    }
    err = cudaGraphDestroy(graph);
    CHECK_CUDA_ERROR(err, "Failed to destroy original graph");
    err = cudaGraphDestroy(clonedGraph);
    CHECK_CUDA_ERROR(err, "Failed to destroy cloned graph");
}
