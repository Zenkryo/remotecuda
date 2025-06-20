#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGraphExternalSemaphoresSignalNode){
    cudaGraph_t graph;
    cudaError_t err = cudaGraphCreate(&graph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create graph");
    cudaGraphNode_t signalNode;
    cudaExternalSemaphoreSignalNodeParams signalParams = {0};
    signalParams.extSemArray = nullptr;
    signalParams.paramsArray = nullptr;
    signalParams.numExtSems = 0;
    err = cudaGraphAddExternalSemaphoresSignalNode(&signalNode, graph, nullptr, 0, &signalParams);
    if(err == cudaSuccess) {
        cudaExternalSemaphoreSignalNodeParams retrievedParams;
        err = cudaGraphExternalSemaphoresSignalNodeGetParams(signalNode, &retrievedParams);
        CHECK_CUDA_ERROR(err, "Failed to get external semaphores signal node parameters");
        err = cudaGraphExternalSemaphoresSignalNodeSetParams(signalNode, &signalParams);
        CHECK_CUDA_ERROR(err, "Failed to set external semaphores signal node parameters");
    }
    err = cudaGraphDestroy(graph);
    CHECK_CUDA_ERROR(err, "Failed to destroy graph");
}
