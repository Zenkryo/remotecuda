#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGraphMemAllocNode){
    cudaGraph_t graph;
    cudaError_t err = cudaGraphCreate(&graph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create graph");
    cudaGraphNode_t allocNode;
    cudaMemAllocNodeParams allocParams = {};
    allocParams.poolProps.allocType = cudaMemAllocationTypePinned;
    allocParams.poolProps.location.type = cudaMemLocationTypeDevice;
    allocParams.poolProps.location.id = 0;
    allocParams.bytesize = 1024;
    allocParams.dptr = nullptr;
    err = cudaGraphAddMemAllocNode(&allocNode, graph, nullptr, 0, &allocParams);
    if(err == cudaSuccess) {
        cudaMemAllocNodeParams retrievedParams;
        err = cudaGraphMemAllocNodeGetParams(allocNode, &retrievedParams);
        CHECK_CUDA_ERROR(err, "Failed to get memory allocation node parameters");
        ASSERT_EQ(retrievedParams.bytesize, allocParams.bytesize) << "Allocation size mismatch";
    }
    err = cudaGraphDestroy(graph);
    CHECK_CUDA_ERROR(err, "Failed to destroy graph");
}
