#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val)
void checkCudaError(cudaError_t result, const char *func) {
    if(result != cudaSuccess) {
        printf("%s failed with error: %s (%d)\n", func, cudaGetErrorString(result), result);
    }
}

int main() {
    cudaError_t result;
    void *dptr;
    void *hptr;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t memcpyNode;

    // Allocate host memory
    hptr = malloc(1024);
    if(!hptr) {
        printf("Failed to allocate host memory\n");
        return -1;
    }

    // Allocate device memory
    result = cudaMalloc(&dptr, 1024);
    CHECK_CUDA_ERROR(result);
    if(result != cudaSuccess) {
        free(hptr);
        return -1;
    }
    printf("Allocated device pointer: %p\n", dptr);

    // Create a CUDA graph
    result = cudaGraphCreate(&graph, 0);
    CHECK_CUDA_ERROR(result);
    if(result != cudaSuccess) {
        cudaFree(dptr);
        free(hptr);
        return -1;
    }

    // Add memory copy node to graph
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr.ptr = hptr;
    copyParams.srcPtr.pitch = 1024;
    copyParams.srcPtr.xsize = 1024;
    copyParams.srcPtr.ysize = 1;
    copyParams.dstPtr.ptr = dptr;
    copyParams.dstPtr.pitch = 1024;
    copyParams.dstPtr.xsize = 1024;
    copyParams.dstPtr.ysize = 1;
    copyParams.extent.width = 1024;
    copyParams.extent.height = 1;
    copyParams.extent.depth = 1;
    copyParams.kind = cudaMemcpyHostToDevice;

    result = cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &copyParams);
    CHECK_CUDA_ERROR(result);
    if(result != cudaSuccess) {
        cudaFree(dptr);
        free(hptr);
        cudaGraphDestroy(graph);
        return -1;
    }
    printf("Added memory copy node to graph\n");

    // Instantiate the graph
    result = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    CHECK_CUDA_ERROR(result);
    if(result != cudaSuccess) {
        cudaFree(dptr);
        free(hptr);
        cudaGraphDestroy(graph);
        return -1;
    }
    printf("Instantiated graph\n");

    // Launch the graph
    result = cudaGraphLaunch(graphExec, NULL);
    CHECK_CUDA_ERROR(result);
    if(result != cudaSuccess) {
        cudaFree(dptr);
        free(hptr);
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
        return -1;
    }
    printf("Launched graph\n");

    // Synchronize to ensure graph execution is complete
    result = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(result);
    if(result != cudaSuccess) {
        cudaFree(dptr);
        free(hptr);
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
        return -1;
    }
    printf("Graph execution completed\n");

    // Cleanup
    cudaFree(dptr);
    free(hptr);
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);

    return 0;
}
