#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGraphNodeOperations){
    cudaGraph_t parentGraph;
    cudaError_t err = cudaGraphCreate(&parentGraph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create parent graph");
    cudaGraph_t childGraph;
    err = cudaGraphCreate(&childGraph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create child graph");
    cudaGraphNode_t childNode;
    void *kernelArgs[] = {NULL};
    cudaKernelNodeParams kernelParams = {0};
    kernelParams.func = (void *)test_kernel;
    kernelParams.gridDim = dim3(1, 1, 1);
    kernelParams.blockDim = dim3(1, 1, 1);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = NULL;
    err = cudaGraphAddKernelNode(&childNode, childGraph, NULL, 0, &kernelParams);
    CHECK_CUDA_ERROR(err, "Failed to add kernel node to child graph");
    cudaGraphNode_t childGraphNode;
    err = cudaGraphAddChildGraphNode(&childGraphNode, parentGraph, NULL, 0, childGraph);
    CHECK_CUDA_ERROR(err, "Failed to add child graph node");
    cudaGraphExec_t execGraph;
    err = cudaGraphInstantiate(&execGraph, parentGraph, NULL, NULL, 0);
    CHECK_CUDA_ERROR(err, "Failed to instantiate graph");
    err = cudaGraphExecChildGraphNodeSetParams(execGraph, childGraphNode, childGraph);
    CHECK_CUDA_ERROR(err, "Failed to set child graph parameters");
    err = cudaGraphDestroyNode(childGraphNode);
    CHECK_CUDA_ERROR(err, "Failed to destroy child graph node");
    err = cudaGraphExecDestroy(execGraph);
    CHECK_CUDA_ERROR(err, "Failed to destroy executable graph");
    err = cudaGraphDestroy(parentGraph);
    CHECK_CUDA_ERROR(err, "Failed to destroy parent graph");
    err = cudaGraphDestroy(childGraph);
    CHECK_CUDA_ERROR(err, "Failed to destroy child graph");
}
