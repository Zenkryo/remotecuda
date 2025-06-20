#include "common.h"

TEST_F(CudaRuntimeApiTest, CudaGraphOperations) {
    cudaError_t err;
    const int N = 1024;
    size_t size = N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    for(int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    float *d_A, *d_B, *d_C;
    err = cudaMalloc(&d_A, size);
    CHECK_CUDA_ERROR(err, "cudaMalloc d_A failed");
    err = cudaMalloc(&d_B, size);
    CHECK_CUDA_ERROR(err, "cudaMalloc d_B failed");
    err = cudaMalloc(&d_C, size);
    CHECK_CUDA_ERROR(err, "cudaMalloc d_C failed");
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
    cudaGraph_t graph;
    err = cudaGraphCreate(&graph, 0);
    CHECK_CUDA_ERROR(err, "cudaGraphCreate failed");
    cudaGraphNode_t memcpyNodeA, memcpyNodeB, kernelNode;
    cudaMemcpy3DParms memcpyParamsA = {0};
    memcpyParamsA.srcPtr = make_cudaPitchedPtr((void *)h_A, size, N, 1);
    memcpyParamsA.dstPtr = make_cudaPitchedPtr((void *)d_A, size, N, 1);
    memcpyParamsA.extent = make_cudaExtent(size, 1, 1);
    memcpyParamsA.kind = cudaMemcpyHostToDevice;
    err = cudaGraphAddMemcpyNode(&memcpyNodeA, graph, NULL, 0, &memcpyParamsA);
    CHECK_CUDA_ERROR(err, "cudaGraphAddMemcpyNode A failed");
    cudaMemcpy3DParms memcpyParamsB = {0};
    memcpyParamsB.srcPtr = make_cudaPitchedPtr((void *)h_B, size, N, 1);
    memcpyParamsB.dstPtr = make_cudaPitchedPtr((void *)d_B, size, N, 1);
    memcpyParamsB.extent = make_cudaExtent(size, 1, 1);
    memcpyParamsB.kind = cudaMemcpyHostToDevice;
    err = cudaGraphAddMemcpyNode(&memcpyNodeB, graph, NULL, 0, &memcpyParamsB);
    CHECK_CUDA_ERROR(err, "cudaGraphAddMemcpyNode B failed");
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
    err = cudaGraphAddKernelNode(&kernelNode, graph, NULL, 0, &kernelParams);
    CHECK_CUDA_ERROR(err, "cudaGraphAddKernelNode failed");
    err = cudaGraphAddDependencies(graph, &memcpyNodeA, &kernelNode, 1);
    CHECK_CUDA_ERROR(err, "cudaGraphAddDependencies A failed");
    err = cudaGraphAddDependencies(graph, &memcpyNodeB, &kernelNode, 1);
    CHECK_CUDA_ERROR(err, "cudaGraphAddDependencies B failed");
    cudaGraphExec_t graphExec;
    err = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    CHECK_CUDA_ERROR(err, "cudaGraphInstantiate failed");
    err = cudaGraphLaunch(graphExec, 0);
    CHECK_CUDA_ERROR(err, "cudaGraphLaunch failed");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err, "cudaMemcpy failed");
    for(int i = 0; i < N; i++) {
        if(fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Verification failed at index %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}
