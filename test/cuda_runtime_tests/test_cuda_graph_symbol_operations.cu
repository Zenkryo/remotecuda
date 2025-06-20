#include "common.h"

__device__ float d_symbol[1024];

TEST_F(CudaRuntimeApiTest, CudaGraphSymbolOperations) {
    cudaError_t err;
    const int N = 1024;
    size_t size = N * sizeof(float);
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    for(int i = 0; i < N; i++) {
        h_input[i] = (float)i;
        h_output[i] = 0.0f;
    }
    float *d_buffer;
    err = cudaMalloc(&d_buffer, size);
    CHECK_CUDA_ERROR(err, "cudaMalloc d_buffer failed");
    int device;
    cudaGetDevice(&device);
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    int computeCapability = major * 10 + minor;
    if(computeCapability < 35) {
        fprintf(stderr, "Device compute capability %d.%d is less than 3.5\n", major, minor);
        cudaFree(d_buffer);
        free(h_input);
        free(h_output);
        exit(EXIT_FAILURE);
    }
    void *symbol_addr;
    err = cudaGetSymbolAddress(&symbol_addr, d_symbol);
    CHECK_CUDA_ERROR(err, "cudaGetSymbolAddress failed");
    cudaGraph_t graph;
    err = cudaGraphCreate(&graph, 0);
    CHECK_CUDA_ERROR(err, "cudaGraphCreate failed");
    cudaGraphNode_t memcpyToSymbolNode, memcpyFromSymbolNode;
    err = cudaGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph, NULL, 0, d_symbol, h_input, size, 0, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err, "cudaGraphAddMemcpyNodeToSymbol failed");
    err = cudaGraphAddMemcpyNodeFromSymbol(&memcpyFromSymbolNode, graph, &memcpyToSymbolNode, 1, d_buffer, d_symbol, size, 0, cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR(err, "cudaGraphAddMemcpyNodeFromSymbol failed");
    cudaGraphExec_t graphExec;
    err = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    CHECK_CUDA_ERROR(err, "cudaGraphInstantiate failed");
    err = cudaGraphLaunch(graphExec, 0);
    CHECK_CUDA_ERROR(err, "cudaGraphLaunch failed");
    err = cudaMemcpy(h_output, d_buffer, size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err, "cudaMemcpy failed");
    for(int i = 0; i < N; i++) {
        if(fabs(h_input[i] - h_output[i]) > 1e-5) {
            fprintf(stderr, "Verification failed at index %d: expected %f, got %f\n", i, h_input[i], h_output[i]);
            cudaGraphExecDestroy(graphExec);
            cudaGraphDestroy(graph);
            cudaFree(d_buffer);
            free(h_input);
            free(h_output);
            exit(EXIT_FAILURE);
        }
    }
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_buffer);
    free(h_input);
    free(h_output);
}
