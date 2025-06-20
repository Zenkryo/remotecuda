#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGraphExternalSemaphoresWaitNode){
    cudaGraph_t graph;
    cudaError_t err = cudaGraphCreate(&graph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create graph");
    cudaGraphNode_t waitNode;
    cudaExternalSemaphoreWaitNodeParams waitParams = {0};
    waitParams.extSemArray = nullptr;
    waitParams.paramsArray = nullptr;
    waitParams.numExtSems = 0;
    err = cudaGraphAddExternalSemaphoresWaitNode(&waitNode, graph, nullptr, 0, &waitParams);
    if(err == cudaSuccess) {
        cudaExternalSemaphoreWaitNodeParams retrievedParams;
        err = cudaGraphExternalSemaphoresWaitNodeGetParams(waitNode, &retrievedParams);
        CHECK_CUDA_ERROR(err, "Failed to get external semaphores wait node parameters");
        err = cudaGraphExternalSemaphoresWaitNodeSetParams(waitNode, &waitParams);
        CHECK_CUDA_ERROR(err, "Failed to set external semaphores wait node parameters");
    }
    err = cudaGraphDestroy(graph);
    CHECK_CUDA_ERROR(err, "Failed to destroy graph");
}
