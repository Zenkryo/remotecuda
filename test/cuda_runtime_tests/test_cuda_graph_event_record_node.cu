#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGraphEventRecordNode){
    cudaGraph_t graph;
    cudaError_t err = cudaGraphCreate(&graph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create graph");
    cudaEvent_t event;
    err = cudaEventCreate(&event);
    CHECK_CUDA_ERROR(err, "Failed to create event");
    cudaGraphNode_t eventRecordNode;
    err = cudaGraphAddEventRecordNode(&eventRecordNode, graph, nullptr, 0, event);
    CHECK_CUDA_ERROR(err, "Failed to add event record node");
    cudaEvent_t retrievedEvent;
    err = cudaGraphEventRecordNodeGetEvent(eventRecordNode, &retrievedEvent);
    CHECK_CUDA_ERROR(err, "Failed to get event");
    ASSERT_EQ(retrievedEvent, event) << "Event mismatch";
    cudaEvent_t newEvent;
    err = cudaEventCreate(&newEvent);
    CHECK_CUDA_ERROR(err, "Failed to create new event");
    err = cudaGraphEventRecordNodeSetEvent(eventRecordNode, newEvent);
    CHECK_CUDA_ERROR(err, "Failed to set event record node event");
    err = cudaGraphDestroy(graph);
    CHECK_CUDA_ERROR(err, "Failed to destroy graph");
    err = cudaEventDestroy(event);
    CHECK_CUDA_ERROR(err, "Failed to destroy event");
    err = cudaEventDestroy(newEvent);
    CHECK_CUDA_ERROR(err, "Failed to destroy new event");
}
