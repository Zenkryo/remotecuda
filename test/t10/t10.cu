#include <cuda_runtime.h>
#include <stdio.h>

// Simple CUDA kernel
__global__ void addKernel(int *c, const int *a, const int *b) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}

int main() {
    // Error checking macro
    #define CUDA_CHECK(call) { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    }

    // Initialize data
    const int N = 1024;
    size_t size = N * sizeof(int);
    
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);
    
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device memory
    int *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Create CUDA graph
    cudaGraph_t graph;
    CUDA_CHECK(cudaGraphCreate(&graph, 0));

    // Create kernel node parameters
    cudaKernelNodeParams kernelNodeParams = {0};
    void* kernelArgs[] = { &d_c, &d_a, &d_b };
    
    kernelNodeParams.func = (void*)addKernel;
    kernelNodeParams.gridDim = dim3(N/256, 1, 1);
    kernelNodeParams.blockDim = dim3(256, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = kernelArgs;
    kernelNodeParams.extra = NULL;

    // Add kernel node to graph
    cudaGraphNode_t kernelNode;
    CUDA_CHECK(cudaGraphAddKernelNode(&kernelNode, graph, NULL, 0, &kernelNodeParams));

    // Create child graph
    cudaGraph_t childGraph;
    CUDA_CHECK(cudaGraphCreate(&childGraph, 0));

    // Add child graph node
    cudaGraphNode_t childGraphNode;
    CUDA_CHECK(cudaGraphAddChildGraphNode(&childGraphNode, graph, &kernelNode, 1, childGraph));

    // Instantiate and launch graph
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    CUDA_CHECK(cudaGraphLaunch(graphExec, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify result
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Verification failed at index %d: %d != %d + %d\n", 
                   i, h_c[i], h_a[i], h_b[i]);
            break;
        }
    }
    printf("Verification completed successfully\n");

    // Cleanup
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(childGraph));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
