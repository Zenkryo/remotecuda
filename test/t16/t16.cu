#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void checkCudaError(cudaError_t err, const char *msg) {
    if(err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Vector size
    const int N = 1024;
    size_t size = N * sizeof(float);

    // Host arrays
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize input arrays
    for(int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Device arrays
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc(&d_C, size), "cudaMalloc d_C failed");

    // Check compute capability
    int device;
    cudaGetDevice(&device);
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    int computeCapability = major * 10 + minor;
    if(computeCapability < 35) {
        fprintf(stderr, "Device compute capability %d.%d is less than 3.5\n", major, minor);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        exit(EXIT_FAILURE);
    }
    printf("Device compute capability: %d.%d\n", major, minor);

    // Create CUDA graph
    cudaGraph_t graph;
    checkCudaError(cudaGraphCreate(&graph, 0), "cudaGraphCreate failed");

    // Graph nodes
    cudaGraphNode_t memcpyNodeA, memcpyNodeB, kernelNode;

    // Add memcpy node for A (host to device)
    cudaMemcpy3DParms memcpyParamsA = {0};
    memcpyParamsA.srcPtr = make_cudaPitchedPtr((void *)h_A, size, N, 1);
    memcpyParamsA.dstPtr = make_cudaPitchedPtr((void *)d_A, size, N, 1);
    memcpyParamsA.extent = make_cudaExtent(size, 1, 1);
    memcpyParamsA.kind = cudaMemcpyHostToDevice;
    checkCudaError(cudaGraphAddMemcpyNode(&memcpyNodeA, graph, NULL, 0, &memcpyParamsA), "cudaGraphAddMemcpyNode A failed");

    // Add memcpy node for B (host to device)
    cudaMemcpy3DParms memcpyParamsB = {0};
    memcpyParamsB.srcPtr = make_cudaPitchedPtr((void *)h_B, size, N, 1);
    memcpyParamsB.dstPtr = make_cudaPitchedPtr((void *)d_B, size, N, 1);
    memcpyParamsB.extent = make_cudaExtent(size, 1, 1);
    memcpyParamsB.kind = cudaMemcpyHostToDevice;
    checkCudaError(cudaGraphAddMemcpyNode(&memcpyNodeB, graph, NULL, 0, &memcpyParamsB), "cudaGraphAddMemcpyNode B failed");

    // Add kernel node
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    void *kernelArgs[] = {(void *)&d_A, (void *)&d_B, (void *)&d_C, (void *)&N};
    cudaKernelNodeParams kernelParams = {0};
    kernelParams.func = (void *)vectorAdd;
    kernelParams.gridDim = grid;
    kernelParams.blockDim = block;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = NULL;
    checkCudaError(cudaGraphAddKernelNode(&kernelNode, graph, NULL, 0, &kernelParams), "cudaGraphAddKernelNode failed");

    // Add dependencies: kernelNode depends on memcpyNodeA and memcpyNodeB
    checkCudaError(cudaGraphAddDependencies(graph, &memcpyNodeA, &kernelNode, 1), "cudaGraphAddDependencies A failed");
    checkCudaError(cudaGraphAddDependencies(graph, &memcpyNodeB, &kernelNode, 1), "cudaGraphAddDependencies B failed");

    // Instantiate and launch the graph
    cudaGraphExec_t graphExec;
    checkCudaError(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0), "cudaGraphInstantiate failed");
    checkCudaError(cudaGraphLaunch(graphExec, 0), "cudaGraphLaunch failed");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "cudaMemcpy failed");

    // Verify result
    for(int i = 0; i < N; i++) {
        if(fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Verification failed at index %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Vector addition completed successfully!\n");

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
