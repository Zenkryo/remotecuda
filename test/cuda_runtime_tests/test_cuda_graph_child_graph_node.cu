#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGraphChildGraphNode){
    cudaGraph_t parentGraph = nullptr;
    cudaGraph_t childGraph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cudaError_t err;
    err = cudaGraphCreate(&parentGraph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create parent graph");
    ASSERT_NE(parentGraph, nullptr);
    cudaGraphNode_t kernelNode;
    void *kernelArgs[] = {};
    cudaKernelNodeParams nodeParams = {};
    nodeParams.func = (void *)test_kernel;
    nodeParams.gridDim = dim3(1, 1, 1);
    nodeParams.blockDim = dim3(1, 1, 1);
    nodeParams.sharedMemBytes = 0;
    nodeParams.kernelParams = kernelArgs;
    nodeParams.extra = nullptr;
    err = cudaGraphAddKernelNode(&kernelNode, parentGraph, nullptr, 0, &nodeParams);
    CHECK_CUDA_ERROR(err, "Failed to add kernel node to parent graph");
    err = cudaGraphCreate(&childGraph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create child graph");
    ASSERT_NE(childGraph, nullptr);
    cudaGraphNode_t childGraphNode;
    err = cudaGraphAddChildGraphNode(&childGraphNode, parentGraph, &kernelNode, 1, childGraph);
    if(err == cudaErrorNotSupported) {
        cudaGraphDestroy(childGraph);
        cudaGraphDestroy(parentGraph);
        GTEST_SKIP() << "Child graph nodes not supported on this device";
    }
    CHECK_CUDA_ERROR(err, "Failed to add child graph node");
    ASSERT_NE(childGraphNode, nullptr);
    cudaGraph_t retrievedGraph;
    err = cudaGraphChildGraphNodeGetGraph(childGraphNode, &retrievedGraph);
    CHECK_CUDA_ERROR(err, "Failed to get child graph");
    ASSERT_NE(retrievedGraph, nullptr);
    err = cudaGraphInstantiate(&graphExec, parentGraph, nullptr, nullptr, 0);
    CHECK_CUDA_ERROR(err, "Failed to instantiate graph");
    err = cudaGraphLaunch(graphExec, nullptr);
    CHECK_CUDA_ERROR(err, "Failed to launch graph");
    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err, "Failed to synchronize device");
    if(graphExec != nullptr) {
        err = cudaGraphExecDestroy(graphExec);
        CHECK_CUDA_ERROR(err, "Failed to destroy graph execution");
    }
    if(parentGraph != nullptr) {
        err = cudaGraphDestroy(parentGraph);
        CHECK_CUDA_ERROR(err, "Failed to destroy parent graph");
    }
    if(childGraph != nullptr) {
        err = cudaGraphDestroy(childGraph);
        CHECK_CUDA_ERROR(err, "Failed to destroy child graph");
    }
}
