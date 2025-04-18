#include <cuda_runtime.h>
#include <stdio.h>

// Simple CUDA kernel to initialize array
__global__ void initKernel(int *data, int value, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N) {
        data[idx] = value;
    }
}

int main() {
// Error checking macro
#define CUDA_CHECK(call)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
    {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          \
        cudaError_t err = call;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                \
        if(err != cudaSuccess) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                                                                                                                                                                                                                                                                                                                                                                                                                         \
            exit(1);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           \
        }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \
    }

    // Initialize data
    const int N = 1024;
    size_t size = N * sizeof(int);

    // Allocate host memory for verification
    int *h_data = (int *)malloc(size);
    if(!h_data) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(1);
    }

    // Allocate device memory
    int *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    // Create CUDA graph
    cudaGraph_t graph;
    CUDA_CHECK(cudaGraphCreate(&graph, 0));

    // Create kernel node parameters
    cudaKernelNodeParams kernelNodeParams = {0};
    int value = 42; // Value to initialize array
    void *kernelArgs[] = {&d_data, (void *)&value, (void *)&N};

    kernelNodeParams.func = (void *)initKernel;
    kernelNodeParams.gridDim = dim3((N + 255) / 256, 1, 1);
    kernelNodeParams.blockDim = dim3(256, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = kernelArgs;
    kernelNodeParams.extra = NULL;

    // Add kernel node to graph
    cudaGraphNode_t kernelNode;
    CUDA_CHECK(cudaGraphAddKernelNode(&kernelNode, graph, NULL, 0, &kernelNodeParams));

    // Create a stream for the graph
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Instantiate and launch graph
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Launch the graph
    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));

    // Wait for graph execution to complete
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Free the device memory after graph execution
    CUDA_CHECK(cudaFree(d_data));

    // Verify memory is freed by attempting to access it
    cudaError_t err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    if(err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer) {
        printf("Memory successfully freed\n");
    } else {
        printf("Unexpected error or memory not freed: %s\n", cudaGetErrorString(err));
    }

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    free(h_data);

    return 0;
}
