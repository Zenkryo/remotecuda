#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGraphEventWaitNode){
    cudaGraph_t graph;
    cudaError_t err = cudaGraphCreate(&graph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create graph");
    cudaEvent_t event;
    err = cudaEventCreate(&event);
    CHECK_CUDA_ERROR(err, "Failed to create event");
    cudaGraphNode_t eventWaitNode;
    err = cudaGraphAddEventWaitNode(&eventWaitNode, graph, nullptr, 0, event);
    CHECK_CUDA_ERROR(err, "Failed to add event wait node");
    cudaEvent_t retrievedEvent;
    err = cudaGraphEventWaitNodeGetEvent(eventWaitNode, &retrievedEvent);
    CHECK_CUDA_ERROR(err, "Failed to get event");
    ASSERT_EQ(retrievedEvent, event) << "Event mismatch";
    cudaEvent_t newEvent;
    err = cudaEventCreate(&newEvent);
    CHECK_CUDA_ERROR(err, "Failed to create new event");
    err = cudaGraphEventWaitNodeSetEvent(eventWaitNode, newEvent);
    CHECK_CUDA_ERROR(err, "Failed to set event wait node event");
    err = cudaGraphDestroy(graph);
    CHECK_CUDA_ERROR(err, "Failed to destroy graph");
    err = cudaEventDestroy(event);
    CHECK_CUDA_ERROR(err, "Failed to destroy event");
    err = cudaEventDestroy(newEvent);
    CHECK_CUDA_ERROR(err, "Failed to destroy new event");
}
