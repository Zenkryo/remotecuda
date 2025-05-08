#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ float d_symbol[1024];

int main() {
    const int N = 1024;
    size_t size = N * sizeof(float);

    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    for(int i = 0; i < N; i++) {
        h_input[i] = (float)i;
        h_output[i] = 0.0f;
    }

    float *d_buffer;
    cudaMalloc(&d_buffer, size);

    // int device;
    // cudaGetDevice(&device);
    // int major, minor;
    // cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    // cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    // int computeCapability = major * 10 + minor;
    // if(computeCapability < 35) {
    //     fprintf(stderr, "Device compute capability %d.%d is less than 3.5\n", major, minor);
    //     cudaFree(d_buffer);
    //     free(h_input);
    //     free(h_output);
    //     exit(EXIT_FAILURE);
    // }
    void *symbol_addr;
    cudaGetSymbolAddress(&symbol_addr, d_symbol);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaGraphNode_t memcpyToSymbolNode, memcpyFromSymbolNode;

    // Add memcpy node to copy from host to device symbol
    cudaGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph, NULL, 0, d_symbol, h_input, size, 0, cudaMemcpyHostToDevice);

    // Add memcpy node to copy from device symbol to device buffer
    cudaGraphAddMemcpyNodeFromSymbol(&memcpyFromSymbolNode, graph, &memcpyToSymbolNode, 1, d_buffer, d_symbol, size, 0, cudaMemcpyDeviceToDevice);

    // Instantiate and launch the graph
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);

    // Copy result from device buffer to host
    cudaMemcpy(h_output, d_buffer, size, cudaMemcpyDeviceToHost);

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
    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_buffer);
    free(h_input);
    free(h_output);
}
