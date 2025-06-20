#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGraphMemcpyNode){
    cudaGraph_t graph;
    cudaError_t err = cudaGraphCreate(&graph, 0);
    CHECK_CUDA_ERROR(err, "Failed to create graph");
    void *hostPtr = malloc(1024);
    ASSERT_NE(hostPtr, nullptr);
    void *devPtr;
    err = cudaMalloc(&devPtr, 1024);
    CHECK_CUDA_ERROR(err, "Failed to allocate device memory");
    cudaGraphNode_t memcpyNode;
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(hostPtr, 1024, 1024, 1);
    copyParams.dstPtr = make_cudaPitchedPtr(devPtr, 1024, 1024, 1);
    copyParams.extent = make_cudaExtent(1024, 1, 1);
    copyParams.kind = cudaMemcpyHostToDevice;
    err = cudaGraphAddMemcpyNode(&memcpyNode, graph, nullptr, 0, &copyParams);
    if(err == cudaErrorNotSupported) {
        GTEST_SKIP() << "Graph memcpy node not supported on this device";
    }
    CHECK_CUDA_ERROR(err, "Failed to add memcpy node");
    cudaMemcpy3DParms retrievedParams;
    err = cudaGraphMemcpyNodeGetParams(memcpyNode, &retrievedParams);
    CHECK_CUDA_ERROR(err, "Failed to get memcpy node parameters");
    ASSERT_EQ(retrievedParams.srcPtr.ptr, copyParams.srcPtr.ptr) << "Source pointer mismatch";
    ASSERT_EQ(retrievedParams.dstPtr.ptr, copyParams.dstPtr.ptr) << "Destination pointer mismatch";
    copyParams.srcPtr = make_cudaPitchedPtr(devPtr, 1024, 1024, 1);
    copyParams.dstPtr = make_cudaPitchedPtr(hostPtr, 1024, 1024, 1);
    copyParams.extent = make_cudaExtent(1024, 1, 1);
    copyParams.kind = cudaMemcpyDeviceToHost;
    err = cudaGraphMemcpyNodeSetParams(memcpyNode, &copyParams);
    if(err == cudaErrorNotSupported) {
        GTEST_SKIP() << "Graph memcpy node parameter setting not supported on this device";
    }
    CHECK_CUDA_ERROR(err, "Failed to set memcpy node parameters");
    err = cudaGraphDestroy(graph);
    CHECK_CUDA_ERROR(err, "Failed to destroy graph");
    err = cudaFree(devPtr);
    CHECK_CUDA_ERROR(err, "Failed to free device memory");
    free(hostPtr);
}
