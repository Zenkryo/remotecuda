#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGraphHostNode){
    cudaGraph_t graph;
    cudaError_t err = cudaGraphCreate(&graph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create graph");
    cudaGraphNode_t hostNode;
    cudaHostNodeParams hostParams = {0};
    hostParams.fn = [](void *userData) {};
    hostParams.userData = nullptr;
    err = cudaGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams);
    CHECK_CUDA_ERROR(err, "Failed to add host node");
    cudaHostNodeParams retrievedParams;
    err = cudaGraphHostNodeGetParams(hostNode, &retrievedParams);
    CHECK_CUDA_ERROR(err, "Failed to get host node parameters");
    ASSERT_EQ(retrievedParams.fn, hostParams.fn) << "Function pointer mismatch";
    hostParams.fn = [](void *userData) {};
    err = cudaGraphHostNodeSetParams(hostNode, &hostParams);
    CHECK_CUDA_ERROR(err, "Failed to set host node parameters");
    err = cudaGraphDestroy(graph);
    CHECK_CUDA_ERROR(err, "Failed to destroy graph");
}
