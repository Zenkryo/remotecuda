#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Global device symbol
__device__ float d_symbol[1024];

void checkCudaError(cudaError_t err, const char *msg) {
    if(err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Data size
    const int N = 1024;
    size_t size = N * sizeof(float);

    // Host arrays
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    // Initialize input array
    for(int i = 0; i < N; i++) {
        h_input[i] = (float)i;
        h_output[i] = 0.0f;
    }

    // Device buffer
    float *d_buffer;
    checkCudaError(cudaMalloc(&d_buffer, size), "cudaMalloc d_buffer failed");

    // Check compute capability
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
    printf("Device compute capability: %d.%d\n", major, minor);

    // Get symbol address
    void *symbol_addr;
    checkCudaError(cudaGetSymbolAddress(&symbol_addr, d_symbol), "cudaGetSymbolAddress failed");

    // Create CUDA graph
    cudaGraph_t graph;
    checkCudaError(cudaGraphCreate(&graph, 0), "cudaGraphCreate failed");

    // Graph nodes
    cudaGraphNode_t memcpyToSymbolNode, memcpyFromSymbolNode;

    // Add memcpy node to copy from host to device symbol
    checkCudaError(cudaGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph, NULL, 0, d_symbol, h_input, size, 0, cudaMemcpyHostToDevice), "cudaGraphAddMemcpyNodeToSymbol failed");

    // Add memcpy node to copy from device symbol to device buffer
    checkCudaError(cudaGraphAddMemcpyNodeFromSymbol(&memcpyFromSymbolNode, graph, &memcpyToSymbolNode, 1, d_buffer, d_symbol, size, 0, cudaMemcpyDeviceToDevice), "cudaGraphAddMemcpyNodeFromSymbol failed");

    // Instantiate and launch the graph
    cudaGraphExec_t graphExec;
    checkCudaError(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0), "cudaGraphInstantiate failed");
    checkCudaError(cudaGraphLaunch(graphExec, 0), "cudaGraphLaunch failed");

    // Copy result from device buffer to host
    checkCudaError(cudaMemcpy(h_output, d_buffer, size, cudaMemcpyDeviceToHost), "cudaMemcpy failed");

    // Verify result
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
    printf("Symbol copy completed successfully!\n");

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_buffer);
    free(h_input);
    free(h_output);

    return 0;
}
