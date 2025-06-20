#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGraphKernelNode){
    cudaGraph_t graph;
    cudaError_t err = cudaGraphCreate(&graph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create graph");
    cudaGraphNode_t kernelNode;
    void *kernelArgs[] = {};
    cudaKernelNodeParams nodeParams = {};
    nodeParams.func = (void *)test_kernel;
    nodeParams.gridDim = dim3(1, 1, 1);
    nodeParams.blockDim = dim3(1, 1, 1);
    nodeParams.sharedMemBytes = 0;
    nodeParams.kernelParams = kernelArgs;
    nodeParams.extra = nullptr;
    err = cudaGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &nodeParams);
    CHECK_CUDA_ERROR(err, "Failed to add kernel node");
    cudaKernelNodeParams retrievedParams;
    err = cudaGraphKernelNodeGetParams(kernelNode, &retrievedParams);
    CHECK_CUDA_ERROR(err, "Failed to get kernel node parameters");
    ASSERT_EQ(retrievedParams.func, nodeParams.func) << "Kernel function mismatch";
    ASSERT_EQ(retrievedParams.gridDim.x, nodeParams.gridDim.x) << "Grid dimension mismatch";
    ASSERT_EQ(retrievedParams.blockDim.x, nodeParams.blockDim.x) << "Block dimension mismatch";
    nodeParams.gridDim = dim3(2, 2, 1);
    err = cudaGraphKernelNodeSetParams(kernelNode, &nodeParams);
    CHECK_CUDA_ERROR(err, "Failed to set kernel node parameters");
    err = cudaGraphDestroy(graph);
    CHECK_CUDA_ERROR(err, "Failed to destroy graph");
}
