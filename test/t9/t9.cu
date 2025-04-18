#include <cuda_runtime.h>
#include <stdio.h>

// Simple CUDA kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Error handling
    cudaError_t err = cudaSuccess;
    
    // Vector size
    const int N = 1024;
    size_t size = N * sizeof(float);
    
    // Host vectors
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Create CUDA memory pool
    cudaMemPool_t memPool;
    cudaMemPoolProps poolProps = {};
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.location.id = 0; // Current device
    poolProps.location.type = cudaMemLocationTypeDevice;
    
    err = cudaMemPoolCreate(&memPool, &poolProps);
    if (err != cudaSuccess) {

        fprintf(stderr, "cudaMemPoolCreate failed: %d %s\n", err, cudaGetErrorString(err));
        return 1;
    }
    
    // Device vectors
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    
    // Allocate memory from pool
    err = cudaMallocFromPoolAsync(&d_A, size, memPool, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocFromPoolAsync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMallocFromPoolAsync(&d_B, size, memPool, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocFromPoolAsync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMallocFromPoolAsync(&d_C, size, memPool, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocFromPoolAsync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Copy inputs to device
    err = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, 0>>>(d_A, d_B, d_C, N);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Copy result back
    err = cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Verify result
    for (int i = 0; i < N; i++) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            return 1;
        }
    }
    
    printf("Vector addition completed successfully!\n");
    
    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaMemPoolDestroy(memPool);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
