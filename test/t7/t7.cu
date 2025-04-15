#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define BLOCK_SIZE 16

// CUDA kernel for matrix addition
__global__ void matrixAdd(float *A, float *B, float *C, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n && j < n) {
        C[i * n + j] = A[i * n + j] + B[i * n + j];
    }
}

int main() {
    float *h_A, *h_B, *h_C; // Host matrices
    float *d_A, *d_B, *d_C; // Device matrices
    size_t size = N * N * sizeof(float);
    
    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // Initialize host matrices
    for(int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Register host memory with CUDA
    cudaHostRegister(h_A, size, cudaHostRegisterDefault);
    cudaHostRegister(h_B, size, cudaHostRegisterDefault);
    cudaHostRegister(h_C, size, cudaHostRegisterDefault);
    
    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Set up grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    matrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify result (simple check)
    for(int i = 0; i < 10; i++) {
        int idx = rand() % (N * N);
        float expected = h_A[idx] + h_B[idx];
        if(fabs(h_C[idx] - expected) > 1e-5) {
            printf("Verification failed at index %d!\n", idx);
            break;
        }
    }
    
    // Cleanup
    cudaHostUnregister(h_A);
    cudaHostUnregister(h_B);
    cudaHostUnregister(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("Matrix addition completed!\n");
    return 0;
}
